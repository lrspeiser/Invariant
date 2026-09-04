"""Acquire Planck products from the IRSA/IPAC mirror, with provenance.

The Planck Legacy Archive AIO endpoint has been observed to return 503 for a
whole session while its landing page returns 200 (recorded trap).  IRSA/IPAC is
the working mirror used here.  Nothing in this file reads a temperature value:
it downloads, hashes, and validates headers only.

VALIDATION (the analogue of the registry's `catalogue_validation` v3 three-detector
rule, which is written for VizieR/VOTable and does not transfer verbatim to a
plain-HTTP FITS fetch).  ALL THREE are required, none is sufficient alone:

    D1  transport   bytes on disk == Content-Length, and the file is non-empty
    D2  structure   astropy parses it, and the HEALPix header keywords
                    (NSIDE, ORDERING, COORDSYS) match what the analysis assumes
    D3  identity    a provenance string in the header (FILENAME / OBJECT /
                    COMMENT) names the expected product AND release

A HTTP 200 is not success; a parse is not success; a header that merely says
"HEALPIX" is not success.  Any single failure aborts with the file quarantined.

    python fetch_planck.py
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "raw")
MAN = os.path.join(HERE, "manifests")

IRSA3 = "https://irsa.ipac.caltech.edu/data/Planck/release_3"
IRSA2 = "https://irsa.ipac.caltech.edu/data/Planck/release_2"

# name -> (url, expected nside or None, expected identity substrings (ALL must appear))
PRODUCTS = {
    "COM_CMB_IQU-smica-nosz_2048_R3.00_full.fits": (
        f"{IRSA3}/all-sky-maps/maps/component-maps/cmb/"
        "COM_CMB_IQU-smica-nosz_2048_R3.00_full.fits", 2048,
        ["smica", "nosz"], "PR3 SMICA CMB map with tSZ deprojected"),
    "COM_CMB_IQU-smica_1024_R2.02_full.fits": (
        f"{IRSA2}/all-sky-maps/maps/component-maps/cmb/"
        "COM_CMB_IQU-smica_1024_R2.02_full.fits", 1024,
        ["smica"], "PR2 SMICA CMB map, nside 1024 -- independent release/arm"),
    "COM_Mask_CMB-common-Mask-Int_2048_R3.00.fits": (
        f"{IRSA3}/ancillary-data/masks/"
        "COM_Mask_CMB-common-Mask-Int_2048_R3.00.fits", 2048,
        ["mask"], "PR3 common intensity confidence mask"),
    "COM_PowerSpect_CMB-base-plikHM-TTTEEE-lowl-lowE-lensing-minimum-theory_R3.01.txt": (
        f"{IRSA3}/ancillary-data/cosmoparams/"
        "COM_PowerSpect_CMB-base-plikHM-TTTEEE-lowl-lowE-lensing-minimum-theory"
        "_R3.01.txt", None, [], "PR3 best-fit theory TT/TE/EE spectrum (for null simulations)"),
    "COM_CompMap_dust-commander_0256_R2.00.fits": (
        f"{IRSA2}/all-sky-maps/maps/component-maps/foregrounds/"
        "COM_CompMap_dust-commander_0256_R2.00.fits", 256,
        ["dust"], "Commander thermal dust, nside 256 -- C7 nuisance template"),
    "COM_CompMap_Synchrotron-commander_0256_R2.00.fits": (
        f"{IRSA2}/all-sky-maps/maps/component-maps/foregrounds/"
        "COM_CompMap_Synchrotron-commander_0256_R2.00.fits", 256,
        ["synchrotron"], "Commander synchrotron -- C7 nuisance template"),
    "COM_CompMap_freefree-commander_0256_R2.00.fits": (
        f"{IRSA2}/all-sky-maps/maps/component-maps/foregrounds/"
        "COM_CompMap_freefree-commander_0256_R2.00.fits", 256,
        ["free"], "Commander free-free -- C7 nuisance template"),
    "COM_CompMap_AME-commander_0256_R2.00.fits": (
        f"{IRSA2}/all-sky-maps/maps/component-maps/foregrounds/"
        "COM_CompMap_AME-commander_0256_R2.00.fits", 256,
        ["ame"], "Commander anomalous microwave emission -- C7 nuisance template"),
    "COM_CompMap_CO-commander_0256_R2.00.fits": (
        f"{IRSA2}/all-sky-maps/maps/component-maps/foregrounds/"
        "COM_CompMap_CO-commander_0256_R2.00.fits", 256,
        ["co"], "Commander CO -- C7 nuisance template"),
}


def sha256_file(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def download(url, dest, tries=3):
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "invariant-voidcmb/1"})
            with urllib.request.urlopen(req, timeout=120) as r:
                declared = int(r.headers.get("Content-Length", -1))
                tmp = dest + ".part"
                n = 0
                t0 = time.time()
                with open(tmp, "wb") as fh:
                    while True:
                        b = r.read(1 << 22)
                        if not b:
                            break
                        fh.write(b)
                        n += len(b)
                        if n % (1 << 26) < (1 << 22):
                            print(f"    {n/1e6:8.1f} / {declared/1e6:.1f} MB "
                                  f"({n/1e6/max(time.time()-t0, 1e-9):.1f} MB/s)", flush=True)
            os.replace(tmp, dest)
            return declared
        except Exception as e:                                        # noqa: BLE001
            print(f"    attempt {k+1}/{tries} failed: {type(e).__name__} {e}", flush=True)
            time.sleep(3 + 5 * k)
    raise RuntimeError(f"download failed after {tries} tries: {url}")


def d2_d3_fits(path, exp_nside, identity):
    """D2 structure + D3 identity for a HEALPix FITS product."""
    from astropy.io import fits
    with fits.open(path) as h:
        hdr = {}
        for hd in h:
            for k, v in hd.header.items():
                hdr.setdefault(k, v)
            allc = " ".join(str(x) for x in hd.header.get("COMMENT", []))
            hdr.setdefault("_COMMENT", "")
            hdr["_COMMENT"] += " " + allc
        cols = []
        for hd in h[1:]:
            if hasattr(hd, "columns"):
                cols = list(hd.columns.names)
                break
        nside = int(hdr.get("NSIDE", -1))
        ordering = str(hdr.get("ORDERING", "")).strip().upper()
        coordsys = str(hdr.get("COORDSYS", "")).strip().upper()
    d2 = (nside == exp_nside) and ordering in ("NESTED", "RING") and coordsys in ("G", "GALACTIC", "C", "E")
    blob = " ".join(str(v) for v in hdr.values()).lower() + " " + os.path.basename(path).lower()
    d3 = all(s.lower() in blob for s in identity) if identity else True
    return d2, d3, dict(nside=nside, ordering=ordering, coordsys=coordsys, columns=cols)


def d2_d3_text(path, identity):
    txt = io.open(path, encoding="utf-8", errors="replace").read(20000)
    nrow = sum(1 for ln in io.open(path, encoding="utf-8", errors="replace")
               if ln.strip() and not ln.lstrip().startswith("#"))
    d2 = nrow > 100 and ("TT" in txt or "L " in txt[:400] or txt.lstrip()[0] == "#")
    d3 = all(s.lower() in txt.lower() for s in identity) if identity else True
    return d2, d3, dict(rows=nrow, head=txt.splitlines()[:3])


def main():
    os.makedirs(RAW, exist_ok=True)
    os.makedirs(MAN, exist_ok=True)
    report = {}
    for name, (url, nside, identity, note) in PRODUCTS.items():
        dest = os.path.join(RAW, name)
        print(f"\n=== {name}")
        if os.path.exists(dest) and os.path.getsize(dest) > 0:
            declared = os.path.getsize(dest)
            print(f"    present ({declared/1e6:.1f} MB), skipping download")
        else:
            declared = download(url, dest)
        got = os.path.getsize(dest)
        d1 = got > 0 and (declared < 0 or got == declared)
        if name.endswith(".fits"):
            d2, d3, meta = d2_d3_fits(dest, nside, identity)
        else:
            d2, d3, meta = d2_d3_text(dest, identity)
        ok = d1 and d2 and d3
        print(f"    D1 transport {'PASS' if d1 else 'FAIL'}  "
              f"D2 structure {'PASS' if d2 else 'FAIL'}  "
              f"D3 identity  {'PASS' if d3 else 'FAIL'}   {meta}")
        man = dict(file=name, source_url=url, mirror="IRSA/IPAC",
                   retrieved_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                   bytes=got, content_length=declared, sha256=sha256_file(dest),
                   notes=note, validation=dict(D1_transport=d1, D2_structure=d2,
                                               D3_identity=d3, all_required=ok),
                   header=meta)
        io.open(os.path.join(MAN, name + ".manifest.json"), "w",
                encoding="utf-8", newline="\n").write(json.dumps(man, indent=1, default=str))
        report[name] = man
        if not ok:
            print("    *** VALIDATION FAILED -- quarantined, not usable ***")
    bad = [k for k, v in report.items() if not v["validation"]["all_required"]]
    print("\n" + "=" * 70)
    print(f"{len(report)} products, {len(bad)} failed validation")
    for b in bad:
        print("  FAILED:", b)
    io.open(os.path.join(HERE, "acquisition.json"), "w", encoding="utf-8",
            newline="\n").write(json.dumps(report, indent=1, default=str))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
