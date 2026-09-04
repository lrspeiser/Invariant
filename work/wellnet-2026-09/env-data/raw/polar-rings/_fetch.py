"""Fetch helper for the polar-ring acquisition lane.

Writes the raw upstream bytes unmodified plus a sibling <name>.manifest.json
carrying: source URL, UTC ISO-8601 retrieval timestamp, SHA-256, byte size,
row count, column names with units, and the exact query issued.

VizieR guard (recorded programme failure mode): VizieR answers HTTP 200 with a
generic HTML page when -source= names a catalogue that does not exist.  Every
VizieR fetch here asserts (a) the body is not HTML, (b) the '#INFO request='
line echoes the catalogue identifier we asked for, and (c) at least one
'#Column' header line is present.
"""
import hashlib
import io
import json
import os
import sys
import urllib.parse
from datetime import datetime, timezone

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
UA = {"User-Agent": "wellnet-2026-09 polar-ring acquisition (research; contact leonard@horizon3.net)"}


def utcnow():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def write_manifest(path, **kw):
    mpath = path + ".manifest.json" if not path.endswith(".manifest.json") else path
    base = os.path.splitext(os.path.basename(path))[0]
    doc = {"file": os.path.basename(path)}
    doc.update(kw)
    with open(mpath, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2)
    return mpath


def get(url, params=None, timeout=300):
    r = requests.get(url, params=params, headers=UA, timeout=timeout)
    r.raise_for_status()
    return r


def parse_vizier_tsv(text):
    """Return (columns, datarows, info_request) from a VizieR asu-tsv body.

    VizieR asu-tsv layout:
      '#Column  <name>  (<fmt>)  <description>  [ucd=...]'  header block
      then a header line of tab-separated names, a units line, a '---' rule,
      then the data rows.  Multiple tables are separated by blank lines and a
      new '#Table' block.
    """
    cols = []
    info_request = ""
    for line in text.split("\n"):
        if line.startswith("#INFO\trequest="):
            info_request = line.split("=", 1)[1].split("\t")[0]
        if line.startswith("#Column"):
            parts = line.split("\t")
            # '#Column', name, '(fmt)', description, '[ucd=..]'
            nm = parts[1].strip() if len(parts) > 1 else ""
            desc = parts[3].strip() if len(parts) > 3 else ""
            fmt = parts[2].strip() if len(parts) > 2 else ""
            cols.append({"name": nm, "format": fmt, "description": desc})
    rows = []
    in_data = False
    for line in text.split("\n"):
        if line.startswith("#") or not line.strip():
            continue
        if set(line.replace("\t", "")) <= set("-") and "-" in line:
            in_data = True
            continue
        if in_data:
            rows.append(line)
    return cols, rows, info_request


def vizier(cat, outname, table=None, extra=None, out_all=True):
    """Download one VizieR catalogue/table as TSV with all columns."""
    params = {"-source": cat if table is None else "%s/%s" % (cat, table),
              "-out.max": "unlimited"}
    if out_all:
        params["-out.all"] = ""
    if extra:
        params.update(extra)
    url = "https://vizier.cds.unistra.fr/viz-bin/asu-tsv"
    r = get(url, params=params)
    text = r.text
    low = text.lstrip()[:200].lower()
    assert not low.startswith("<!doctype") and not low.startswith("<html"), \
        "VizieR returned HTML for %s -- catalogue probably does not exist" % cat
    cols, rows, info_request = parse_vizier_tsv(text)
    assert cols, "VizieR returned no #Column headers for %s" % cat
    # echo-back assertion: the identifier we asked for must appear in the
    # request URL VizieR reports back to us
    assert urllib.parse.quote(cat, safe="") in info_request.replace("%2F", "/").replace("/", "%2F") \
        or cat in urllib.parse.unquote(info_request), \
        "VizieR did not echo back %s (got %s)" % (cat, info_request)
    path = os.path.join(HERE, outname)
    with open(path, "wb") as f:
        f.write(r.content)
    write_manifest(
        path,
        source_url=r.url,
        catalogue=cat,
        vizier_table=table,
        retrieved_utc=utcnow(),
        sha256=sha256_bytes(r.content),
        bytes=len(r.content),
        row_count=len(rows),
        column_count=len(cols),
        columns=cols,
        query="GET %s" % r.url,
        vizier_echoed_request=urllib.parse.unquote(info_request),
        extraction="Raw VizieR asu-tsv response, unmodified. -out.all -out.max=unlimited.",
    )
    print("OK  %-46s rows=%-6d cols=%-3d bytes=%d" % (outname, len(rows), len(cols), len(r.content)))
    return path, rows, cols


def plain(url, outname, note="", row_count=None, columns=None, extraction=""):
    r = get(url)
    path = os.path.join(HERE, outname)
    with open(path, "wb") as f:
        f.write(r.content)
    if row_count is None:
        try:
            row_count = len([l for l in r.text.split("\n") if l.strip()])
        except Exception:
            row_count = None
    write_manifest(path, source_url=url, retrieved_utc=utcnow(),
                   sha256=sha256_bytes(r.content), bytes=len(r.content),
                   row_count=row_count, columns=columns or [],
                   query="GET %s" % url, note=note,
                   extraction=extraction or "Raw upstream response, unmodified.")
    print("OK  %-46s bytes=%d lines=%s" % (outname, len(r.content), row_count))
    return path
