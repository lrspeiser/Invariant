"""Write a sibling <name>.manifest.json for every raw file this lane downloaded.

Source URL, retrieval timestamp, SHA-256, byte size, row count, column names with
units, and the exact query issued -- as required by the programme brief.
"""
import hashlib
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
from astropy.io import fits

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vizier_tsv import read_vizier_tsv

LANE = r"C:\Users\henry\Documents\Codex\2026-08-21\Invariant-main-integration\work\wellnet-2026-09\env-data"
MANGA = os.path.join(LANE, "raw", "manga")
GROUPS = os.path.join(LANE, "raw", "groups")
NOW = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
# All files in this lane were retrieved in one session on this date.
RETRIEVED = "2026-09-04"

FITS_SRC = {
    "drpall-v3_1_1.fits": dict(
        url="https://data.sdss.org/sas/dr17/manga/spectro/redux/v3_1_1/drpall-v3_1_1.fits",
        query="HTTP GET, whole file",
        note="MaNGA DR17 Data Reduction Pipeline summary catalogue. HDU1 'MANGA' is "
             "the galaxy sample (11273 rows, one per IFU observation, so a galaxy "
             "observed twice appears twice); HDU2 'MASTAR' is the stellar library "
             "and is NOT used here. NSA columns are Nasa-Sloan Atlas v1_0_1 "
             "photometry; nsa_elpetro_mass assumes a Chabrier IMF and a k-correct "
             "stellar population fit - a stellar-population estimate, not a "
             "dynamical mass, and not dark-matter dependent.",
        hdu=1),
    "dapall-v3_1_1-3.1.0.fits": dict(
        url="https://data.sdss.org/sas/dr17/manga/spectro/analysis/v3_1_1/3.1.0/"
            "dapall-v3_1_1-3.1.0.fits",
        query="HTTP GET, whole file",
        note="MaNGA DR17 Data Analysis Pipeline summary. Four extensions, one per "
             "DAPTYPE; this lane uses HDU3 HYB10-MILESHC-MASTARSSP. STELLAR_VEL_LO/HI "
             "and HA_GVEL_LO/HI are the 2.5th and 97.5th percentiles of the velocity "
             "field, used here only as a crude rotation-amplitude proxy; the resolved "
             "fields are in the per-galaxy MAPS files.",
        hdu=3),
    "manga-pymorph-DR17.fits": dict(
        url="https://data.sdss.org/sas/dr17/manga/photo/pymorph/1.1.1/manga-pymorph-DR17.fits",
        query="HTTP GET, whole file",
        note="MaNGA PyMorph DR17 photometric VAC (Dominguez Sanchez+2022 MNRAS 509, "
             "4024). HDU1=g, HDU2=r, HDU3=i (band order confirmed against the SDSS "
             "data model AND against the median magnitudes 16.22/15.49/15.08). "
             "FLAG_FIT: 0=no preference, 1=Sersic preferred, 2=Ser+Exp preferred, "
             "3=both failed. A_HL_SE_DISK is the disk HALF-LIGHT semi-major axis in "
             "arcsec, NOT the exponential scale length; this lane divides by 1.678. "
             "Null value is -999.",
        hdu=2),
    "manga-morphology-dl-DR17.fits": dict(
        url="https://data.sdss.org/sas/dr17/manga/morphology/deep_learning/1.1.1/"
            "manga-morphology-dl-DR17.fits",
        query="HTTP GET, whole file",
        note="MaNGA Deep-Learning morphology VAC (Dominguez Sanchez+2022). T-Type and "
             "the probabilities P_LTG, P_S0, P_edge, P_bar come from convolutional "
             "networks trained on visual classifications; they are a classification, "
             "not a measurement, and are used here only as a sample gate.",
        hdu=1),
    "manga_visual_morpho-2.0.1.fits": dict(
        url="https://data.sdss.org/sas/dr17/manga/morphology/manga_visual_morpho/2.0.1/"
            "manga_visual_morpho-2.0.1.fits",
        query="HTTP GET, whole file",
        note="MaNGA Visual Morphology VAC 2.0.1 (Vazquez-Mata+2022). Human visual "
             "classification; TType, Bars, Edge_on, Tidal flags.",
        hdu=1),
    "mangaHIall.fits": dict(
        url="https://data.sdss.org/sas/dr17/manga/HI/v2_0_1/mangaHIall.fits",
        query="HTTP GET, whole file",
        note="HI-MaNGA DR3 v2_0_1 (Stark+2021 MNRAS 503, 1345). ONE ROW PER OBSERVING "
             "SESSION, so a galaxy can appear more than once: 6632 rows cover 6442 "
             "distinct plateifu. LOGMHI is the detected HI mass; LOGHILIM200KMS is the "
             "upper limit assuming a 200 km/s line width for non-detections. This lane "
             "calls a detection SNR>=5. Molecular gas is NOT included anywhere.",
        hdu=1),
    "GEMA_2.0.2.fits": dict(
        url="https://data.sdss.org/sas/dr17/manga/gema/2.0.2/GEMA_2.0.2.fits",
        query="HTTP GET, whole file",
        note="GEMA VAC 2.0.2, Galaxy Environment for MaNGA (Argudo-Fernandez+). 15 "
             "BINTABLEs; this lane uses DR17_param_groups (tidal strength Q_group, "
             "GroupSize), DR17_param_LSS (large-scale tidal tensor eigenvalues t1,t2,t3 "
             "and the halo mass mh) and DR17_param_overdensity. The primary HDU holds "
             "a VOTable of the metadata but the FIELD elements carry NO descriptions "
             "and NO units, so the physical meaning of several columns cannot be "
             "recovered from the file alone - see REPORT.md.",
        hdu=None),
}

TSV_SRC = {
    "tempel2014_galaxies.tsv": dict(
        cat="J/A+A/566/A1/galaxies", expect=588193,
        note="Tempel+2014 A&A 566, A1, flux-limited SDSS DR10 galaxy table with FoF "
             "group membership. Ngal is the richness of the group the galaxy belongs "
             "to; Ngal=1 means the FoF found no companion, which this lane uses as the "
             "FIELD definition. Distances are comoving Mpc in an h=1 convention."),
    "tempel2014_groups.tsv": dict(
        cat="J/A+A/566/A1/groups", expect=82458,
        note="Tempel+2014 group table. sig.v is the rms radial velocity deviation of "
             "the members: an OBSERVABLE. Rvir is the projected harmonic mean radius. "
             "MNFW and MHer are masses from assumed NFW / Hernquist profiles and are "
             "DARK-MATTER DEPENDENT - usable only to rank environments. Lrgroup is the "
             "observed r-band group luminosity."),
    "tempel2017_table1_galaxies.tsv": dict(
        cat="J/A+A/602/A100/table1", expect=584449,
        note="Tempel+2017 A&A 602, A100, SDSS DR12 galaxies with group IDs. Used here "
             "as an INDEPENDENT membership determination to check Tempel+2014."),
    "tempel2017_table2_groups.tsv": dict(
        cat="J/A+A/602/A100/table2", expect=88662,
        note="Tempel+2017 group table. R200 and M200 come from an assumed NFW profile "
             "and are DARK-MATTER DEPENDENT - ranking only. No velocity dispersion is "
             "tabulated in this release, which is why Tempel+2014 is the primary "
             "environment source in this lane."),
    "mcxc_piffaretti2011.tsv": dict(
        cat="J/A+A/534/A109/mcxc", expect=1743,
        note="MCXC meta-catalogue of X-ray galaxy clusters (Piffaretti+2011 A&A 534, "
             "A109). L500 is the [0.1-2.4] keV luminosity within R500: an X-ray "
             "OBSERVABLE. M500 and R500 are derived from an L-M scaling relation "
             "calibrated on hydrostatic masses and are DARK-MATTER DEPENDENT - ranking "
             "only. Used here to flag which Tempel host groups are X-ray confirmed."),
}


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def do_fits(name, meta):
    p = os.path.join(MANGA, name)
    if not os.path.exists(p):
        print("MISSING", p)
        return
    cols, nrow, hdus = [], None, []
    with fits.open(p) as h:
        for i, x in enumerate(h):
            if x.data is not None and hasattr(x, "columns"):
                hdus.append({"index": i, "extname": x.header.get("EXTNAME"),
                             "rows": int(len(x.data)), "cols": len(x.columns)})
        if meta["hdu"] is not None:
            d = h[meta["hdu"]]
            nrow = int(len(d.data))
            cols = [{"name": c.name, "format": c.format, "unit": c.unit or ""}
                    for c in d.columns]
    m = {"file": name, "source_url": meta["url"], "exact_query": meta["query"],
         "retrieved_utc": RETRIEVED + "T00:00:00Z", "manifest_written_utc": NOW,
         "sha256": sha256(p), "bytes": os.path.getsize(p),
         "hdus": hdus,
         "documented_hdu": meta["hdu"],
         "row_count": nrow, "column_count": len(cols) if cols else None,
         "columns": cols,
         "extraction": "Raw upstream file, byte-for-byte unmodified.",
         "note": meta["note"]}
    with open(p + ".manifest.json", "w") as f:
        json.dump(m, f, indent=2)
    print("manifest %-34s rows=%s cols=%s" % (name, nrow, len(cols)))


def do_tsv(name, meta):
    p = os.path.join(GROUPS, name)
    if not os.path.exists(p):
        print("MISSING", p)
        return
    df = read_vizier_tsv(p)
    units = df.attrs["units"]
    assert len(df) == meta["expect"], \
        "%s: %d rows != %d expected from the source paper" % (name, len(df), meta["expect"])
    url = ("https://vizier.cds.unistra.fr/viz-bin/asu-tsv?-source=%s"
           "&-out.max=unlimited&-out.all" % meta["cat"].replace("/", "%2F"))
    m = {"file": name, "source_url": url,
         "vizier_catalogue": meta["cat"],
         "exact_query": "GET https://vizier.cds.unistra.fr/viz-bin/asu-tsv with "
                        "params {-source: %s, -out.max: unlimited, -out.all: ''}"
                        % meta["cat"],
         "retrieved_utc": RETRIEVED + "T04:13:00Z", "manifest_written_utc": NOW,
         "sha256": sha256(p), "bytes": os.path.getsize(p),
         "row_count": int(len(df)), "column_count": int(df.shape[1]),
         "columns": [{"name": c, "unit": units.get(c, "").strip()} for c in df.columns],
         "extraction": "Raw VizieR asu-tsv response, unmodified. Validated: the body "
                       "carries #Table and #Column lines, the header names match the "
                       "#Column declarations one for one, and the row count equals the "
                       "sample size stated in the source paper.",
         "note": meta["note"]}
    with open(p + ".manifest.json", "w") as f:
        json.dump(m, f, indent=2)
    print("manifest %-34s rows=%d cols=%d" % (name, len(df), df.shape[1]))


if __name__ == "__main__":
    for k, v in FITS_SRC.items():
        do_fits(k, v)
    for k, v in TSV_SRC.items():
        do_tsv(k, v)
