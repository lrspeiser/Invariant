"""Provenance-preserving downloader for the wellnet-2026-09 cluster-data lane.

Every file gets a sibling <name>.manifest.json with source URL, UTC retrieval
timestamp, SHA-256, byte size, row/column counts, column names, and the exact
query issued.  Nothing is transformed in place: the raw upstream bytes are what
land on disk.
"""
import hashlib
import json
import os
import sys
import datetime
import urllib.request

LANE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

UA = "Mozilla/5.0 (compatible; gravity-lane-acquire/1.0; research)"


def utcnow():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url, dest, timeout=300):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = r.read()
        code = r.status
    with open(dest, "wb") as f:
        f.write(data)
    return code, len(data)


def write_manifest(dest, url, note, extraction, exact_query=None,
                   row_count=None, column_count=None, columns=None, extra=None):
    m = {
        "file": os.path.basename(dest),
        "source_url": url,
        "exact_query": exact_query if exact_query is not None else url,
        "retrieved_utc": utcnow(),
        "sha256": sha256(dest),
        "bytes": os.path.getsize(dest),
        "row_count": row_count,
        "column_count": column_count,
        "columns": columns,
        "extraction": extraction,
        "note": note,
    }
    if extra:
        m.update(extra)
    with open(dest + ".manifest.json", "w", encoding="utf-8") as f:
        json.dump(m, f, indent=2)
    return m


def fits_table_info(path, hdu=1):
    """Return (nrows, ncols, [{name, unit, format}]) for a FITS binary table."""
    from astropy.io import fits
    with fits.open(path) as hl:
        h = hl[hdu]
        cols = [{"name": c.name, "unit": (c.unit or ""), "format": c.format}
                for c in h.columns]
        return h.data.shape[0], len(cols), cols
