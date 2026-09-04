"""VizieR / URL acquisition helper with manifests and the traps this programme has hit.

Traps guarded here:
  * VizieR returns HTTP 200 with a generic page for a nonexistent -source=.
    The reliable check is the ASU `#Name:` echo in the returned header block:
    we assert the echoed catalogue name matches what we asked for.
  * Silent truncation. Every ingest asserts a row count and a column count.
  * Provenance. Every file gets <name>.manifest.json with source URL, UTC
    timestamp, sha256, bytes, rows, columns+units, and the exact query.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import urllib.parse
import urllib.request

LANE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(LANE, "data")
RAW = os.path.join(DATA, "raw")
os.makedirs(RAW, exist_ok=True)

UA = "gravity-programme-acquire/1.0 (research; contact leonard@horizon3.net)"
VIZ = "https://vizier.cds.unistra.fr/viz-bin/asu-tsv"


def _now():
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def fetch(url: str, timeout: int = 180) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def vizier_tsv(source: str, out_max: str = "unlimited", extra: dict | None = None) -> tuple[str, str]:
    """Return (url, text) for a VizieR ASU TSV query on -source=<source>."""
    q = [("-source", source), ("-out", "**"), ("-out.max", out_max)]
    if extra:
        q += list(extra.items())
    url = VIZ + "?" + urllib.parse.urlencode(q, safe="*/+.")
    txt = fetch(url).decode("utf-8", "replace")
    return url, txt


def echo_ok(txt: str, source: str) -> tuple[bool, str]:
    """The anti-trap: VizieR echoes `#Name: <catalogue>` for a real table."""
    names = [ln.split(":", 1)[1].strip() for ln in txt.splitlines()
             if ln.startswith("#Name:")]
    want = source.split("/")
    for n in names:
        if n.replace(" ", "").lower().startswith("/".join(want[:2]).lower()):
            return True, "; ".join(names)
    # accept if the exact source string appears in an echoed name
    for n in names:
        if source.lower() in n.lower():
            return True, "; ".join(names)
    return False, "; ".join(names) if names else "<no #Name: echo>"


def parse_asu(txt: str):
    """Parse a VizieR asu-tsv body into (header_comment, cols, units, rows)."""
    lines = txt.splitlines()
    # data block: after the last '#'-comment run, cols line, units line, ---- line
    body = [ln for ln in lines if not ln.startswith("#")]
    # strip leading blanks
    while body and not body[0].strip():
        body.pop(0)
    if len(body) < 3:
        return "\n".join(l for l in lines if l.startswith("#")), [], [], []
    cols = body[0].split("\t")
    units = body[1].split("\t")
    # separator line of dashes
    k = 2
    if body[k].strip().startswith("-"):
        k += 1
    rows = [ln.split("\t") for ln in body[k:] if ln.strip() and not ln.startswith("#")]
    rows = [r for r in rows if len(r) == len(cols)]
    return "\n".join(l for l in lines if l.startswith("#")), cols, units, rows


def save(name: str, text: str, *, url: str, query: str, cols, units, nrow,
         note: str, extra_manifest: dict | None = None, raw_text: str | None = None):
    path = os.path.join(DATA, name)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    b = text.encode("utf-8")
    man = {
        "file": name,
        "source_url": url,
        "exact_query": query,
        "retrieved_utc": _now(),
        "sha256": hashlib.sha256(b).hexdigest(),
        "bytes": len(b),
        "row_count": nrow,
        "column_count": len(cols),
        "columns": [{"name": c, "unit": u} for c, u in zip(cols, units)],
        "note": note,
    }
    if raw_text is not None:
        rp = os.path.join(RAW, name + ".raw")
        with open(rp, "w", encoding="utf-8", newline="\n") as f:
            f.write(raw_text)
        man["raw_response_file"] = os.path.relpath(rp, DATA).replace("\\", "/")
        man["raw_sha256"] = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()
    if extra_manifest:
        man.update(extra_manifest)
    with open(path + ".manifest.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(man, f, indent=2)
    return path, man
