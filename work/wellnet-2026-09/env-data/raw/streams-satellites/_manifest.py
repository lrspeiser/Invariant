"""Manifest helper for the streams/satellites/counter-rotator acquisition lane.

Usage (import, do not execute downloaded code):

    from _manifest import write_manifest, sha256_of, http_get

Every downloaded file gets a sibling <name>.manifest.json carrying:
source URL, retrieval timestamp (UTC ISO-8601), SHA-256, byte size, row count,
column names with units, and the exact query issued.
"""
import hashlib
import json
import os
import datetime


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def utcnow():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def write_manifest(path, source_url, query, columns, row_count,
                   note=None, extraction=None, retrieved_utc=None,
                   source_file_within_archive=None, extra=None,
                   measurement_or_model=None):
    """columns: list of dicts {"name":..., "unit":...} or list of str."""
    cols = []
    for c in columns:
        if isinstance(c, str):
            cols.append({"name": c, "unit": ""})
        else:
            cols.append({"name": c.get("name"), "unit": c.get("unit", "")})
    man = {
        "file": os.path.basename(path),
        "source_url": source_url,
    }
    if source_file_within_archive:
        man["source_file_within_archive"] = source_file_within_archive
    man["retrieved_utc"] = retrieved_utc or utcnow()
    man["sha256"] = sha256_of(path)
    man["bytes"] = os.path.getsize(path)
    man["row_count"] = row_count
    man["column_count"] = len(cols)
    man["columns"] = cols
    man["query"] = query
    if extraction:
        man["extraction"] = extraction
    if measurement_or_model:
        man["measurement_or_model"] = measurement_or_model
    if note:
        man["note"] = note
    if extra:
        man.update(extra)
    mpath = path + ".manifest.json"
    with open(mpath, "w", encoding="utf-8") as fh:
        json.dump(man, fh, indent=2)
    print("MANIFEST %s  rows=%s cols=%s bytes=%s sha256=%s"
          % (mpath, row_count, len(cols), man["bytes"], man["sha256"][:16]))
    return mpath


def http_get(url, dest, timeout=180, headers=None):
    import requests
    hdr = {"User-Agent": "gravity-research-acquisition/1.0 (academic data acquisition)"}
    if headers:
        hdr.update(headers)
    r = requests.get(url, timeout=timeout, headers=hdr)
    r.raise_for_status()
    with open(dest, "wb") as fh:
        fh.write(r.content)
    print("GET %s -> %s (%d bytes, ctype=%s)"
          % (url, dest, len(r.content), r.headers.get("Content-Type")))
    return r


def assert_vizier_tsv(path, expect_catalog=None, min_rows=1):
    """VizieR returns HTTP 200 + generic HTML for a nonexistent -source=.
    Assert the payload is really a VizieR TSV and echo the catalogue id back."""
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        txt = fh.read()
    if "<html" in txt[:2000].lower() or "<!doctype" in txt[:2000].lower():
        raise AssertionError("VizieR returned HTML, not TSV: %s" % path)
    # CRITICAL: for a nonexistent catalogue VizieR returns HTTP 200 and a
    # well-formed comment header that ECHOES the requested identifier back
    # inside an error line, e.g.
    #   #INFO Error=Table or Catalog not found: J/ApJ/863/L20
    # so echo-checking alone is NOT sufficient. Fail on any Error= line first.
    for line in txt.splitlines():
        if line.startswith("#INFO") and "Error=" in line:
            raise AssertionError("VizieR reported an error for %s: %s"
                                 % (path, line.strip()))
        if not line.startswith("#"):
            break
    lines = [l for l in txt.splitlines()]
    hdr_idx = None
    for i, l in enumerate(lines):
        if l.startswith("#") or not l.strip():
            continue
        hdr_idx = i
        break
    if hdr_idx is None:
        raise AssertionError("No data section found in %s" % path)
    cols = lines[hdr_idx].split("\t")
    # data rows: skip header, unit row, dash row
    data = [l for l in lines[hdr_idx + 3:] if l.strip() and not l.startswith("#")]
    catline = [l for l in lines[:hdr_idx] if "-source" in l or "Resource" in l or "Table" in l]
    print("VIZIER OK %s  ncols=%d nrows=%d" % (path, len(cols), len(data)))
    print("  header: %s" % (cols[:12],))
    for l in catline[:6]:
        print("  meta: %s" % l.strip())
    if expect_catalog:
        if expect_catalog.lower() not in txt.lower():
            raise AssertionError("Catalogue id %r not echoed back in %s" % (expect_catalog, path))
        print("  ECHO OK: catalogue %r present in response" % expect_catalog)
    if len(data) < min_rows:
        raise AssertionError("Only %d rows (< %d) in %s" % (len(data), min_rows, path))
    return cols, data
