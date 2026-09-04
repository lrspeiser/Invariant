"""Acquisition for the eFEDS x HSC potential-depth lane.

Every downloaded artefact gets a sibling <name>.manifest.json carrying the
source URL, the exact query, a UTC timestamp, SHA-256, byte size, row count and
column names with units, per the standing brief.

WHAT THIS SCRIPT DOES *NOT* DO: it does not fetch a per-source HSC shape
catalogue, because there is none to fetch without an account.  The probe is run
here and its HTTP status is recorded in `access_probes.json` so that the claim
in REPORT.md is evidenced rather than asserted.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import ssl
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ACQ = os.path.join(HERE, "acquire")
os.makedirs(ACQ, exist_ok=True)

UA = "Mozilla/5.0 (research; gravity-programme; leonard@horizon3.net)"
CTX = ssl.create_default_context()


def utcnow():
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def get(url, timeout=120):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout, context=CTX) as r:
        return r.status, r.read()


def probe(url):
    """HEAD-ish probe that records the status code, including auth failures."""
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=60, context=CTX) as r:
            return {"url": url, "status": r.status, "bytes": len(r.read(4096)),
                    "note": "read first 4 kB only"}
    except urllib.error.HTTPError as e:
        return {"url": url, "status": e.code, "reason": str(e.reason),
                "www_authenticate": e.headers.get("WWW-Authenticate", "")}
    except Exception as e:                                   # noqa: BLE001
        return {"url": url, "status": None, "error": f"{type(e).__name__}: {e}"}


# --------------------------------------------------------------- VizieR TSV
def parse_vizier_tsv(text):
    """Return (columns, units, rows) from a VizieR asu-tsv payload.

    Silent-extraction guard: VizieR answers HTTP 200 with a generic page for a
    nonexistent -source=, so the caller MUST assert the row count and echo the
    identifier back.
    """
    lines = text.split("\n")
    cols = units = None
    rows = []
    for i, ln in enumerate(lines):
        if ln.startswith("#") or not ln.strip():
            continue
        parts = [p.strip() for p in ln.rstrip("\r").split("\t")]
        if cols is None:
            cols = parts
            continue
        if units is None:
            units = parts
            continue
        if set("".join(parts)) <= set("- "):        # the dashed separator line
            continue
        if len(parts) != len(cols):
            continue
        rows.append(parts)
    return cols, units, rows


def write_manifest(path, meta):
    with open(path + ".manifest.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)


def fetch_vizier(source, outname, expect_rows=None, extra_note=""):
    url = ("https://vizier.cds.unistra.fr/viz-bin/asu-tsv?-source="
           + source.replace("+", "%2B") + "&-out=**&-out.max=unlimited")
    status, raw = get(url)
    out = os.path.join(ACQ, outname)
    with open(out, "wb") as f:
        f.write(raw)
    text = raw.decode("utf-8", "replace")
    cols, units, rows = parse_vizier_tsv(text)
    # echo the identifier back -- guards the generic-page failure mode
    echoed = [ln for ln in text.split("\n") if ln.startswith("#Name:")]
    meta = {
        "file": outname,
        "source_url": url,
        "exact_query": url,
        "http_status": status,
        "vizier_table": source,
        "identifier_echoed_by_server": echoed,
        "retrieved_utc": utcnow(),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
        "row_count": len(rows),
        "column_count": len(cols) if cols else 0,
        "columns": [{"name": c, "unit": u}
                    for c, u in zip(cols or [], units or [])],
        "extraction": "verbatim VizieR asu-tsv response, unmodified",
        "note": extra_note,
    }
    write_manifest(out, meta)
    ok = expect_rows is None or len(rows) == expect_rows
    print(f"   {outname:44s} http {status}  rows {len(rows):5d} "
          f"cols {len(cols or [])}  {'OK' if ok else 'ROW-COUNT MISMATCH'}")
    if not ok:
        raise SystemExit(f"row count {len(rows)} != expected {expect_rows} "
                         f"for {source}")
    return cols, units, rows


def main():
    print("=" * 78)
    print("ACQUISITION -- eFEDS x HSC")
    print("=" * 78)

    # ---------------------------------------------------------------- probes
    print("\n1. Access probes for a RAW per-source shear catalogue")
    probes = [
        "https://hsc-release.mtk.nao.ac.jp/archive/filetree/",
        "https://hsc-release.mtk.nao.ac.jp/archive/filetree/"
        "shape_catalog_y3/catalog_obs_reGaus_public/",
        "https://hsc-release.mtk.nao.ac.jp/archive/filetree/"
        "s16a-shape-catalog/",
        "https://hsc-release.mtk.nao.ac.jp/das_search/pdr3/",
    ]
    results = [probe(u) for u in probes]
    for r in results:
        print(f"   HTTP {str(r.get('status')):5s} {r['url']}")
    with open(os.path.join(ACQ, "access_probes.json"), "w",
              encoding="utf-8") as f:
        json.dump({"probed_utc": utcnow(), "probes": results}, f, indent=1)

    # ------------------------------------------------------------- catalogues
    print("\n2. VizieR catalogues (unauthenticated, work fine)")
    fetch_vizier(
        "J/A+A/661/A11/tablec1", "chiu2022_efeds_tablec1.tsv", 457,
        "Chiu+2022 A&A 661 A11 Table C1.  These are MASS ESTIMATES "
        "(count-rate inferred, weak-lensing calibrated).  Used in this lane "
        "ONLY for sample membership and for a labelled cross-check; NEVER as "
        "the observable, per hard constraint 2 of the standing brief.")
    fetch_vizier(
        "J/A+A/661/A2/table1", "liu2022_efeds_clusters.tsv", None,
        "Liu+2022 eFEDS cluster catalogue -- source of the confirmed-cluster "
        "flags and positions.")
    fetch_vizier(
        "J/ApJ/890/148/table2", "umetsu2020_xxl_hsc_wl.tsv", 136,
        "Umetsu+2020 XXL x HSC.  M200/M500 are NFW-fitted weak-lensing "
        "masses -- forbidden as an observable.  Retained for the frozen-beta "
        "transfer sample definition and for T300kpc, which is a fixed 300 kpc "
        "aperture and NOT core-excised.")
    print("\ndone.")


if __name__ == "__main__":
    main()
