"""Acquire the individual-stream catalogues from VizieR, with manifests.

Every response is validated: HTTP 200 is NOT trusted, the payload must not be
HTML, must not carry an '#INFO Error=' line, must echo the catalogue id, and
must yield the expected row count.
"""
import os
import sys

import requests

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from _manifest import write_manifest, sha256_of, utcnow  # noqa: E402

HDR = {"User-Agent": "gravity-research-acquisition/1.0 (academic data acquisition)"}
URL = "https://vizier.cds.unistra.fr/viz-bin/asu-tsv?-source={cat}&-out.all&-out.max=unlimited"

TARGETS = [
    dict(cat="J/MNRAS/485/4726", stem="stream_koposov2019_orphan_rrl",
         expect_rows=109,
         note="Koposov et al. 2019, MNRAS 485, 4726 -- Orphan-Chenab stream. "
              "VizieR holds ONLY table15: the subset of likely Orphan RR Lyrae "
              "with Gaia DR2 proper motions and RR-Lyrae period-luminosity "
              "DISTANCES. The paper's radial velocities are NOT in this VizieR "
              "record.",
         mm="MEASUREMENT. Positions, Gaia DR2 proper motions and RR-Lyrae "
            "distances. RR-Lyrae distances come from the period-luminosity "
            "relation (a stellar-physics calibration), NOT from any gravitational "
            "model. No dark-matter assumption enters."),
    dict(cat="J/ApJ/914/123", stem="stream_ibata2021_streamfinder_members",
         expect_rows=5960,
         note="Ibata et al. 2021, ApJ 914, 123 -- STREAMFINDER stream member "
              "stars in Gaia EDR3, with per-star stream membership labels. "
              "Columns include heliocentric radial velocities (HRV) compiled "
              "from spectroscopic surveys with a reference column r_HRV.",
         mm="MEASUREMENT for RAJ2000/DEJ2000/plx/pmRA/pmDE/Gmag0/(BP-RP)0/HRV. "
            "CAUTION: the 'dSF' column is the STREAMFINDER-estimated heliocentric "
            "distance. STREAMFINDER detects streams by searching for stars "
            "consistent with a common ORBIT in an ASSUMED Galactic potential, so "
            "dSF is MODEL-DEPENDENT and must be labelled MODEL, not observation. "
            "The astrometry, photometry and HRV columns are independent measurements."),
    dict(cat="J/ApJ/823/157", stem="stream_ishigaki2016_pal5_kinematics",
         expect_rows=151,
         note="Ishigaki et al. 2016, ApJ 823, 157 -- Palomar 5 tidal stream "
              "spectroscopy. Two tables: table4 = FOCAS members with Vlos, "
              "table5 = DEIMOS field stars with HRV. Both are concatenated in "
              "this single VizieR response.",
         mm="MEASUREMENT. Line-of-sight velocities and spectroscopic [Fe/H] "
            "for Pal 5 stream stars. No potential assumed."),
    dict(cat="J/MNRAS/501/2279", stem="stream_vasiliev2021_sagittarius",
         expect_rows=55192,
         note="Vasiliev, Belokurov & Erkal 2021, MNRAS 501, 2279 -- Sagittarius "
              "stream catalogue. Carries Gaia astrometry, distances, "
              "line-of-sight velocities and [Fe/H], plus stream coordinates "
              "Lambda/Beta.",
         mm="MEASUREMENT for RAJ2000/DEJ2000/plx/pmRA/pmDE/Gmag/vLOS/[Fe/H] and "
            "the Lambda/Beta stream coordinates (a pure rotation of the sky "
            "frame). The 'Dist' column is a compiled/adopted heliocentric "
            "distance -- check the Ref column per star for its provenance."),
    dict(cat="J/A+A/635/L3", stem="stream_antoja2020_sgr_pm_map",
         expect_rows=294344,
         note="Antoja et al. 2020, A&A 635, L3 -- all-sky proper-motion map of "
              "the Sagittarius stream from Gaia DR2. Astrometry and photometry "
              "only: NO distances and NO radial velocities in this table.",
         mm="MEASUREMENT. Gaia DR2 positions, proper motions and photometry only."),
]


def parse_vizier_tsv(txt, cat):
    """Return (columns, units, ndata). Fails loudly on VizieR's soft errors."""
    low = txt[:3000].lower()
    if "<html" in low or "<!doctype" in low:
        raise AssertionError("VizieR returned HTML, not TSV, for %s" % cat)
    lines = txt.splitlines()
    for line in lines:
        if line.startswith("#INFO") and "Error=" in line:
            raise AssertionError("VizieR error for %s: %s" % (cat, line.strip()))
        if not line.startswith("#"):
            break
    if cat.lower() not in txt.lower():
        raise AssertionError("catalogue id %s not echoed back" % cat)
    # VizieR TSV: sections separated by blank lines; each section is
    # header / units / dashes / data...
    cols, units, ndata = None, None, 0
    i = 0
    state = 0
    while i < len(lines):
        l = lines[i]
        if l.startswith("#") or not l.strip():
            i += 1
            state = 0
            continue
        if state == 0:
            if cols is None:
                cols = l.split("\t")
            state = 1
        elif state == 1:
            if units is None:
                units = l.split("\t")
            state = 2
        elif state == 2:
            state = 3   # the ---- rule line
        else:
            ndata += 1
        i += 1
    return cols, units, ndata


ok, bad = [], []
for t in TARGETS:
    cat, stem = t["cat"], t["stem"]
    url = URL.format(cat=cat)
    raw = os.path.join(BASE, stem + ".vizier.tsv")
    print("\n=== %s -> %s" % (cat, stem))
    try:
        r = requests.get(url, timeout=600, headers=HDR)
        r.raise_for_status()
        txt = r.text
        cols, units, ndata = parse_vizier_tsv(txt, cat)
        with open(raw, "w", encoding="utf-8", newline="") as fh:
            fh.write(txt)
        print("   rows=%d cols=%d bytes=%d" % (ndata, len(cols), os.path.getsize(raw)))
        print("   cols:", cols[:14])
        if t.get("expect_rows") is not None and ndata != t["expect_rows"]:
            print("   !! ROW COUNT MISMATCH: got %d, expected %d" % (ndata, t["expect_rows"]))
            bad.append((cat, "row mismatch %d != %d" % (ndata, t["expect_rows"])))
        else:
            print("   ASSERT OK: row count %d matches the probe" % ndata)
        write_manifest(
            raw,
            source_url=url,
            query="HTTP GET " + url,
            columns=[{"name": c, "unit": (units[j] if units and j < len(units) else "")}
                     for j, c in enumerate(cols)],
            row_count=ndata,
            extraction="Unmodified VizieR ASU-TSV response, saved verbatim including the "
                       "'#'-prefixed provenance header. Validated: payload is not HTML, "
                       "carries no '#INFO Error=' line, and echoes the catalogue id.",
            measurement_or_model=t["mm"],
            note=t["note"] + (" VizieR catalogue identifier echoed back in the response: %s." % cat),
        )
        ok.append((cat, ndata))
    except Exception as e:
        print("   FAILED:", repr(e)[:300])
        bad.append((cat, repr(e)[:200]))

print("\n================ SUMMARY ================")
for c, n in ok:
    print("  OK   %-20s %8d rows" % (c, n))
for c, e in bad:
    print("  FAIL %-20s %s" % (c, e))
