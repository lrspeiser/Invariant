"""
Write a <name>.manifest.json beside every raw file in this lane, plus a
lane-level MANIFEST.json inventory.  Row/column counts are asserted, not
assumed, and the identifier is echoed back (guard against silent extraction
failures and against servers returning HTTP 200 with the wrong product).
"""
from __future__ import annotations

import gzip
import json
import os
import sys

import numpy as np
import pandas as pd
from astropy.io import fits

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import sha256_file, utc_now, write_manifest

HERE = os.path.dirname(os.path.abspath(__file__))
LANE = os.path.dirname(HERE)
REPO = os.path.abspath(os.path.join(LANE, "..", "..", ".."))
RAW = os.path.join(LANE, "raw")
PRIV = os.path.join(REPO, "work", "private")

RETRIEVED = "2026-09-03"

DESIVAST_BASE = "https://data.desi.lbl.gov/public/dr1/vac/dr1/desivast/v1.0"
LSS_BASE = ("https://data.desi.lbl.gov/public/dr1/survey/catalogs/dr1/LSS/"
            "iron/LSScats/v1.5")
PANTHEON_BASE = ("https://raw.githubusercontent.com/PantheonPlusSH0ES/DataRelease/"
                 "main/Pantheon%2B_Data/4_DISTANCES_AND_COVAR")


def fits_summary(path):
    rows = {}
    cols = {}
    with fits.open(path, memmap=True) as h:
        for hdu in h:
            if getattr(hdu, "columns", None) is not None:
                rows[hdu.name] = int(hdu.header.get("NAXIS2", 0))
                cols[hdu.name] = [
                    {"name": c.name, "format": c.format, "unit": c.unit}
                    for c in hdu.columns]
        prim = {k: (v if isinstance(v, (int, float, str, bool)) else str(v))
                for k, v in h[0].header.items()
                if k not in ("COMMENT", "HISTORY", "")}
    return rows, cols, prim


def main():
    inventory = []

    # ---------------- DESIVAST -----------------------------------------
    published = {}
    sums = os.path.join(RAW, "desivast", "dr1_vac_dr1_desivast_v1.0.sha256sum")
    for line in open(sums):
        h, f = line.split()
        published[f.strip()] = h.strip()

    for fn in sorted(os.listdir(os.path.join(RAW, "desivast"))):
        p = os.path.join(RAW, "desivast", fn)
        if fn.endswith(".manifest.json"):
            continue
        extra = {}
        rows = cols = None
        if fn.endswith(".fits"):
            rows, cols, prim = fits_summary(p)
            extra["fits_hdu_rows"] = rows
            extra["primary_header"] = prim
            extra["cosmology_declared_in_file"] = {
                "OMEGAM": prim.get("OMEGAM"), "HP": prim.get("HP"),
                "ZLIMU": prim.get("ZLIMU"), "DLIMU": prim.get("DLIMU"),
                "METRIC": prim.get("METRIC"), "MAGLIM": prim.get("MAGLIM"),
                "INFILE": prim.get("INFILE")}
        if fn in published:
            actual = sha256_file(p)
            extra["publisher_sha256"] = published[fn]
            extra["publisher_sha256_match"] = bool(actual == published[fn])
            assert actual == published[fn], f"SHA mismatch for {fn}"
        m = write_manifest(
            p, f"{DESIVAST_BASE}/{fn}",
            f"curl -sS -L -o {fn} {DESIVAST_BASE}/{fn}",
            rows, cols,
            notes=("DESI DR1 DESIVAST v1.0 value-added void catalogue "
                   "(Rincon et al. 2025, ApJ 982 38). Verified against the "
                   "publisher's own sha256sum manifest."),
            retrieved=RETRIEVED, extra=extra)
        inventory.append({"path": os.path.relpath(p, LANE), "sha256": m["sha256"],
                          "bytes": m["bytes"], "source": m["source_url"]})

    # ---------------- DESI LSS ------------------------------------------
    lsspub = {}
    s2 = os.path.join(RAW, "desi_lss",
                      "dr1_survey_catalogs_dr1_LSS_iron_LSScats_v1.5.sha256sum")
    for line in open(s2):
        parts = line.split()
        if len(parts) == 2:
            lsspub[os.path.basename(parts[1])] = parts[0]
    for fn in sorted(os.listdir(os.path.join(RAW, "desi_lss"))):
        p = os.path.join(RAW, "desi_lss", fn)
        if fn.endswith(".manifest.json"):
            continue
        extra = {}
        rows = cols = None
        if fn.endswith(".fits"):
            rows, cols, prim = fits_summary(p)
            extra["fits_hdu_rows"] = rows
        if fn in lsspub:
            actual = sha256_file(p)
            extra["publisher_sha256"] = lsspub[fn]
            extra["publisher_sha256_match"] = bool(actual == lsspub[fn])
            assert actual == lsspub[fn], f"SHA mismatch for {fn}"
        m = write_manifest(
            p, f"{LSS_BASE}/{fn}", f"curl -sS -L -o {fn} {LSS_BASE}/{fn}",
            rows, cols,
            notes=("DESI DR1 LSS clustering catalogue v1.5, BGS_BRIGHT. "
                   "Held as an independent cross-check of the galaxy density "
                   "field; the fiducial field is built from the DESIVAST "
                   "GALZONE sample so that field and voids share a sample."),
            retrieved=RETRIEVED, extra=extra)
        inventory.append({"path": os.path.relpath(p, LANE), "sha256": m["sha256"],
                          "bytes": m["bytes"], "source": m["source_url"]})

    # ---------------- Pantheon+ -----------------------------------------
    pdir = os.path.join(RAW, "pantheonplus")
    for fn in sorted(os.listdir(pdir)):
        p = os.path.join(pdir, fn)
        if fn.endswith(".manifest.json"):
            continue
        extra = {}
        rows = cols = None
        if fn.endswith(".dat"):
            df = pd.read_csv(p, sep=r"\s+")
            rows = int(len(df))
            cols = list(df.columns)
            assert rows == 1701, f"Pantheon+ rows {rows} != 1701"
            assert len(cols) == 47, f"Pantheon+ cols {len(cols)} != 47"
        elif fn.endswith(".cov"):
            with open(p) as fh:
                n = int(fh.readline().split()[0])
                vals = np.fromstring(fh.read(), sep=" ")
            assert vals.size == n * n, f"{fn}: {vals.size} != {n*n}"
            rows = n
            cols = [f"N={n} square matrix, row-major, magnitude^2"]
            C = vals.reshape(n, n)
            extra["max_abs_asymmetry"] = float(np.abs(C - C.T).max())
        m = write_manifest(
            p, f"{PANTHEON_BASE}/{fn.replace('+','%2B')}",
            f"curl -sS -L -o '{fn}' {PANTHEON_BASE}/{fn.replace('+','%2B')}",
            rows, cols,
            notes=("Pantheon+ / SH0ES data release (Scolnic et al. 2022, "
                   "Brout et al. 2022). Byte-identical to the copy previously "
                   "frozen at work/private/open-gravity-lane9-pantheonplus-"
                   "c447f0f."),
            retrieved=RETRIEVED, extra=extra)
        inventory.append({"path": os.path.relpath(p, LANE), "sha256": m["sha256"],
                          "bytes": m["bytes"], "source": m["source_url"]})

    # ---------------- reused frozen sources ------------------------------
    reused = [
        (os.path.join(PRIV, "open-gravity-void-source-v2", "cf4-table4.dat.gz"),
         "https://cdsarc.cds.unistra.fr/ftp/cats/J/ApJ/944/94/table4.dat.gz",
         "Cosmicflows-4 group distances, CDS J/ApJ/944/94 table4 "
         "(Tully et al. 2023, ApJ 944 94). 38053 groups.", 38053),
        (os.path.join(PRIV, "open-gravity-void-source-v2", "cf4-ReadMe"),
         "https://cdsarc.cds.unistra.fr/ftp/cats/J/ApJ/944/94/ReadMe",
         "Cosmicflows-4 CDS ReadMe (byte-by-byte column definitions).", None),
        (os.path.join(PRIV, "open-gravity-void-source-v2",
                      "VoidFinder-nsa_v1_0_1_Planck2018_comoving_holes.txt"),
         "https://zenodo.org/records/11043278",
         "SDSS DR7 / NSA VoidFinder void holes, VAST v1.3.1 Zenodo record "
         "11043278 (Douglass et al. 2023, ApJS 265 7), Planck2018 comoving. "
         "INDEPENDENT non-DESI void catalogue.", 39735),
        (os.path.join(PRIV, "open-gravity-void-source-v2",
                      "VoidFinder-nsa_v1_0_1_Planck2018_comoving_maximal.txt"),
         "https://zenodo.org/records/11043278",
         "SDSS DR7 / NSA VoidFinder maximal spheres, VAST v1.3.1.", 1163),
    ]
    for p, url, note, expect in reused:
        if not os.path.exists(p):
            continue
        rows = None
        cols = None
        if p.endswith(".gz"):
            with gzip.open(p, "rt") as fh:
                rows = sum(1 for ln in fh if ln.strip())
        elif p.endswith(".txt"):
            with open(p) as fh:
                head = fh.readline()
                rows = sum(1 for ln in fh if ln.strip())
            cols = head.lstrip("#").split()
        if expect is not None:
            assert rows == expect, f"{os.path.basename(p)}: {rows} != {expect}"
        dest = os.path.join(LANE, "manifests",
                            os.path.basename(p) + ".manifest.json")
        man = {"file": os.path.basename(p), "local_path": os.path.relpath(p, REPO),
               "source_url": url, "retrieved_utc": RETRIEVED,
               "sha256": sha256_file(p), "bytes": os.path.getsize(p),
               "row_count": rows, "columns": cols, "notes": note,
               "reused_from_prior_lane": True}
        with open(dest, "w") as fh:
            json.dump(man, fh, indent=2)
        inventory.append({"path": os.path.relpath(p, REPO), "sha256": man["sha256"],
                          "bytes": man["bytes"], "source": url})

    with open(os.path.join(LANE, "MANIFEST.json"), "w") as fh:
        json.dump({"lane": "work/wellnet-2026-09/void-data",
                   "generated_utc": utc_now(),
                   "n_files": len(inventory), "files": inventory}, fh, indent=2)
    print(f"wrote {len(inventory)} manifests")


if __name__ == "__main__":
    main()
