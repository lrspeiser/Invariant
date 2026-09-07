"""archives.py -- the lane's only routes to an archive, each with its validator.

Three clients, because the three archives fail in three different ways and a
single "did the HTTP call work" check is wrong for all of them.

VIZIER -- `catalogue_validation` v3, THREE detectors, ALL required:
    D1  #Name: echoes the EXACT -source= requested.  A wrong -source= can be
        answered HTTP 200 with an unrelated REAL catalogue (Run AZ was served
        the wrong PAPER; the velocities lane was served Cooper+2013 twelve
        times).
    D2  no CatalogsExamined= anywhere -- that string is the fuzzy-fallback
        signature.
    D3  #Title: matches the expected author/year.
    A zero-row result is NOT an absence until D1-D3 have passed.

TAP -- the NOIRLab Astro Data Lab endpoint answers a malformed or too-heavy
    ADQL query with HTTP 200 whose body is a VOTable carrying
    QUERY_STATUS="ERROR".  Parsed as CSV that reads as zero rows, i.e. a silent
    false negative.  Every response is therefore checked for the XML envelope
    BEFORE it is parsed.  Every query additionally passes through
    guard.check_query, so this lane can only ask an archive to COUNT sources,
    never to hand over a shape.

HTTP -- a bulk file is validated by transport (Content-Length present,
    Accept-Ranges when we intend to range-read) and by identity (magic bytes),
    never by status code alone.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import re
import ssl
import subprocess
import time
import urllib.parse
import urllib.request

import guard

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "raw")
UA = "gravity-goldcluster/1.0 (research; inventory only)"

VIZIER = "https://vizier.cds.unistra.fr/viz-bin/asu-tsv"
TAP = "https://datalab.noirlab.edu/tap/sync"
CTX = ssl.create_default_context()


class ValidationFailure(RuntimeError):
    pass


def _utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# --------------------------------------------------------------------- VizieR
def vizier_url(source, out="**", maxrows="unlimited", extra=None):
    q = [("-source", source), ("-out", out), ("-out.max", str(maxrows))]
    if extra:
        q += list(extra)
    return VIZIER + "?" + urllib.parse.urlencode(q, safe="*/+")


def _curl(url, timeout=600):
    p = subprocess.run(["curl", "-s", "--max-time", str(timeout), "-A", UA, url],
                       capture_output=True)
    if p.returncode != 0:
        raise RuntimeError("curl exit %d" % p.returncode)
    return p.stdout.decode("utf-8", "replace")


def vizier_validate(source, txt, want_title=()):
    """The v3 three-detector rule.  Returns (ok, verdicts)."""
    lines = txt.splitlines()
    d = {}
    head = txt[:800].lower()
    d["not_html"] = "<html" not in head and "<!doctype" not in head
    m = re.search(r"^#INFO\s+Error=(.+)$", txt, re.M)
    d["no_vizier_error"] = m is None
    d["vizier_error"] = m.group(1).strip() if m else None

    names = [l.split(":", 1)[1].strip() for l in lines if l.startswith("#Name:")]
    want = source.rstrip("/")
    parent = "/".join(want.split("/")[:4])
    d["D1_name_echo"] = any(n == want or n == parent for n in names)
    d["names_seen"] = names

    d["D2_no_catalogs_examined"] = not any("CatalogsExamined" in l for l in lines)

    titles = [l.split(":", 1)[1].strip() for l in lines if l.startswith("#Title:")]
    d["titles_seen"] = titles
    joined = " ".join(titles).lower()
    d["D3_title_match"] = (all(t.lower() in joined for t in want_title)
                           if want_title else None)

    sep = [i for i, l in enumerate(lines) if l.startswith("---")]
    d["n_rows"] = (sum(1 for l in lines[sep[-1] + 1:]
                       if l.strip() and not l.startswith("#")) if sep else 0)

    ok = (d["not_html"] and d["no_vizier_error"] and d["D1_name_echo"]
          and d["D2_no_catalogs_examined"] and d["D3_title_match"] is not False)
    return ok, d


def vizier_fetch(source, want_title=(), maxrows="unlimited", save_as=None):
    """Fetch and validate.  Raises ValidationFailure unless every detector passes."""
    url = vizier_url(source, maxrows=maxrows)
    txt = _curl(url)
    ok, d = vizier_validate(source, txt, want_title)
    rec = dict(archive="vizier", source=source, url=url, validated=ok,
               detectors=d, retrieved_utc=_utc(),
               sha256=hashlib.sha256(txt.encode()).hexdigest(),
               bytes=len(txt.encode()))
    if not ok:
        raise ValidationFailure(
            "catalogue_validation v3 FAILED for %s: D1=%s D2=%s D3=%s error=%s"
            % (source, d["D1_name_echo"], d["D2_no_catalogs_examined"],
               d["D3_title_match"], d["vizier_error"]))
    if save_as:
        path = os.path.join(RAW, save_as)
        io.open(path, "w", newline="\n", encoding="utf-8").write(txt)
        rec["path"] = os.path.relpath(path, HERE).replace("\\", "/")
        _manifest(path, rec)
    return txt, rec


def parse_vizier_tsv(txt):
    """Split an asu-tsv payload into (colnames, units, rows)."""
    lines = txt.splitlines()
    sep = [i for i, l in enumerate(lines) if l.startswith("---")]
    if not sep:
        raise ValidationFailure("no '---' separator in the VizieR payload")
    i = sep[-1]
    cols = lines[i - 2].split("\t")
    units = lines[i - 1].split("\t")
    rows = [l.split("\t") for l in lines[i + 1:]
            if l.strip() and not l.startswith("#")]
    return cols, units, rows


# ------------------------------------------------------------------------ TAP
def tap_count(query, timeout=300, retries=3):
    """Send an inventory query.  guard.check_query decides whether it is one."""
    guard.check_query(query)
    body = urllib.parse.urlencode({"REQUEST": "doQuery", "LANG": "ADQL",
                                   "FORMAT": "csv", "QUERY": query}).encode()
    last = None
    for k in range(retries):
        try:
            req = urllib.request.Request(TAP, data=body,
                                         headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=timeout, context=CTX) as r:
                raw = r.read()
            # HTTP 200 is NOT success: a VOTable body is an error report
            if raw.lstrip().startswith(b"<?xml"):
                raise ValidationFailure(
                    "TAP returned an ERROR VOTable under HTTP 200: "
                    + raw[:600].decode("utf-8", "replace"))
            return raw.decode("utf-8", "replace")
        except ValidationFailure:
            raise                        # a real error report; do not retry it
        except Exception as exc:         # noqa: BLE001
            last = exc
            if k == retries - 1:
                raise
            time.sleep(5 * (k + 1))
    raise last                           # pragma: no cover


# ----------------------------------------------------------------------- HTTP
def http_probe(url, magic=None, want_ranges=False):
    """Validate a bulk file by transport and identity WITHOUT downloading it."""
    p = subprocess.run(["curl", "-sIL", "--max-time", "120", "-A", UA, url],
                       capture_output=True)
    hdr = p.stdout.decode("utf-8", "replace")
    size = re.search(r"(?im)^content-length:\s*(\d+)", hdr)
    ranges = re.search(r"(?im)^accept-ranges:\s*bytes", hdr)
    rec = dict(archive="http", url=url, retrieved_utc=_utc(),
               bytes=int(size.group(1)) if size else None,
               accept_ranges=bool(ranges))
    if magic:
        q = subprocess.run(["curl", "-s", "--max-time", "120", "-A", UA,
                            "-r", "0-%d" % (len(magic) - 1), url],
                           capture_output=True)
        rec["magic_ok"] = q.stdout == magic
        rec["magic_seen"] = q.stdout.hex()
    rec["validated"] = (bool(size) and rec.get("magic_ok", True)
                        and (rec["accept_ranges"] or not want_ranges))
    return rec


def http_dir(url):
    """List an archive directory.  Returns the raw text; callers parse it."""
    return _curl(url)


# ------------------------------------------------------------------ manifests
def _manifest(path, rec):
    man = dict(rec)
    man["file"] = os.path.basename(path)
    io.open(path + ".manifest.json", "w", newline="\n",
            encoding="utf-8").write(json.dumps(man, indent=2) + "\n")
    return man
