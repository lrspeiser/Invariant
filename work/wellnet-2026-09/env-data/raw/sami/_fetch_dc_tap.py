"""Acquire SAMI DR3 catalogue tables from the Data Central public TAP service.

No credentials, no registration, no terms accepted: the TAP endpoint
https://datacentral.org.au/vo/tap/sync answers anonymous POSTs.

Raw VOTable bytes are written unmodified; a cleaned TSV and a manifest are
written beside each.  Row counts are asserted against an independent
COUNT(*) query issued before the bulk pull.
"""
import hashlib
import io
import json
import os
import sys
import time
from datetime import datetime, timezone

import requests
from astropy.io.votable import parse

TAP = "https://datacentral.org.au/vo/tap/sync"
OUT = os.path.dirname(os.path.abspath(__file__))

TABLES = [
    "CubeObs",
    "DensityCatDR3",
    "EmissionLine1compDR3",
    "FstarCatClusters",
    "FstarCatGAMA",
    "IndexAperturesDR3",
    "InputCatClustersDR3",
    "InputCatFiller",
    "InputCatGAMADR3",
    "MGEPhotomUnregDR3",
    "samiDR3gaskinPA",
    "samiDR3Stelkin",
    "SSPAperturesDR3",
    "VisualMorphologyDR3",
]

NOTES = {
    "InputCatClustersDR3": (
        "SAMI cluster-region input catalogue (Owers et al. 2017, MNRAS 468, 1824). "
        "R_on_rtwo is projected clustercentric radius / R200 and V_on_sigma is the "
        "line-of-sight velocity offset from the cluster redshift divided by the cluster "
        "velocity dispersion; see the Owers 2017 cluster-property table for the "
        "provenance of R200 (MODEL-DERIVED, virial) versus sigma (OBSERVABLE)."
    ),
    "InputCatGAMADR3": (
        "SAMI GAMA-region (field/group) input catalogue, Bryant et al. 2015, "
        "MNRAS 447, 2857.  Photometry from SDSS DR7; Re from GAMA single-Sersic fits."
    ),
    "samiDR3Stelkin": (
        "Aperture stellar kinematics: velocity dispersion in Sersic-Re, MGE-Re, 3 kpc "
        "and fixed arcsec apertures, plus the spin proxy lambda_R(Re), V/sigma, the "
        "stellar kinematic PA and the kinemetry asymmetry k5/k1."
    ),
    "CubeObs": (
        "One row per SAMI data cube (repeats included).  ISBEST marks the preferred "
        "repeat; the WARN* columns are the DR3 quality flags."
    ),
}


def sha256_bytes(b):
    h = hashlib.sha256()
    h.update(b)
    return h.hexdigest()


def tap_post(query, fmt="votable", maxrec=1000000, timeout=900):
    data = {
        "REQUEST": "doQuery",
        "LANG": "ADQL",
        "FORMAT": fmt,
        "MAXREC": str(maxrec),
        "QUERY": query,
    }
    r = requests.post(TAP, data=data, timeout=timeout)
    r.raise_for_status()
    return r


def count_rows(table):
    q = "SELECT COUNT(*) AS n FROM sami_dr3.%s" % table
    r = tap_post(q, fmt="csv", timeout=300)
    return int(r.text.strip().splitlines()[-1])


def fetch(table):
    expected = count_rows(table)
    query = "SELECT * FROM sami_dr3.%s" % table
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    r = tap_post(query, fmt="votable")
    raw = r.content

    # --- guards: this must be a real VOTable with a TABLEDATA payload -------
    head = raw[:4000].decode("utf-8", "replace")
    assert "<VOTABLE" in head, "%s: response is not a VOTable" % table
    txt = raw.decode("utf-8", "replace")
    assert "<TABLEDATA>" in txt or "<BINARY" in txt, "%s: no data block" % table
    assert (
        'value="ERROR"' not in txt[:8000]
    ), "%s: TAP returned an ERROR status" % table
    if "QUERY_STATUS" in txt and 'value="OVERFLOW"' in txt:
        raise AssertionError("%s: result truncated (OVERFLOW)" % table)

    raw_path = os.path.join(OUT, "sami_dr3_%s.vot" % table)
    with open(raw_path, "wb") as f:
        f.write(raw)

    vot = parse(io.BytesIO(raw))
    tbl = vot.get_first_table()
    arr = tbl.to_table()
    nrow = len(arr)
    assert nrow == expected, "%s: got %d rows, COUNT(*) said %d" % (
        table,
        nrow,
        expected,
    )

    cols = []
    for f_ in tbl.fields:
        cols.append(
            {
                "name": f_.name,
                "unit": (str(f_.unit) if f_.unit else None),
                "datatype": f_.datatype,
                "ucd": f_.ucd,
                "description": (f_.description or None),
            }
        )

    tsv_path = os.path.join(OUT, "sami_dr3_%s.tsv" % table)
    df = arr.to_pandas()
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].apply(
                lambda v: v.decode("utf-8", "replace") if isinstance(v, bytes) else v
            )
    df.to_csv(tsv_path, sep="\t", index=False)
    with open(tsv_path, "rb") as f:
        tsv_bytes = f.read()
    ncheck = sum(1 for _ in open(tsv_path, encoding="utf-8")) - 1
    assert ncheck == nrow, "%s: TSV has %d data lines, table has %d" % (
        table,
        ncheck,
        nrow,
    )

    man = {
        "file": os.path.basename(raw_path),
        "cleaned_file": os.path.basename(tsv_path),
        "source_url": TAP,
        "service": "Data Central (AAO/Macquarie) IVOA TAP v1.0, anonymous access",
        "access_credentials": "none - public anonymous POST, no login, no API key, no terms accepted",
        "query": query,
        "query_parameters": {
            "REQUEST": "doQuery",
            "LANG": "ADQL",
            "FORMAT": "votable",
            "MAXREC": "1000000",
        },
        "count_check_query": "SELECT COUNT(*) AS n FROM sami_dr3.%s" % table,
        "count_check_value": expected,
        "retrieved_utc": ts,
        "sha256": sha256_bytes(raw),
        "bytes": len(raw),
        "cleaned_sha256": sha256_bytes(tsv_bytes),
        "cleaned_bytes": len(tsv_bytes),
        "row_count": nrow,
        "column_count": len(cols),
        "columns": cols,
        "reference": "Croom et al. 2021, MNRAS 505, 991 - the SAMI Galaxy Survey Data Release 3",
        "extraction": "SELECT * over the TAP table, VOTable saved byte-for-byte; TSV is a "
        "direct astropy Table -> pandas dump with no unit conversion, no cut, no join.",
    }
    if table in NOTES:
        man["note"] = NOTES[table]
    with open(os.path.join(OUT, "sami_dr3_%s.vot.manifest.json" % table), "w") as f:
        json.dump(man, f, indent=1)
    print("OK  %-24s rows=%-6d cols=%-4d bytes=%d" % (table, nrow, len(cols), len(raw)))
    return nrow, len(cols)


if __name__ == "__main__":
    want = sys.argv[1:] or TABLES
    summary = {}
    for t in want:
        for attempt in range(3):
            try:
                summary[t] = fetch(t)
                break
            except Exception as e:
                print("FAIL %s attempt %d: %r" % (t, attempt + 1, e))
                time.sleep(5)
        else:
            summary[t] = None
    print(json.dumps(summary, indent=1))
