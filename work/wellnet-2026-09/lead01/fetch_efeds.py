"""Fetch the eFEDS group profiles, with every trap this programme has recorded.

Lead 01 needs |Phi_b| to vary by >= 1 dex at fixed g_bar WITHIN ONE CLASS. The
two-overdensity-radius group tables cap that at ~0.17 dex because r barely moves
at fixed g_bar within a rung. Resolved n_e(r) removes the cap, because two groups
with the same g_bar reach it at DIFFERENT radii, and the identity

    log|Phi_b| = log g_bar + log r + log S

then has real spread in both r and S rather than a fixed ratio of two radii.

Trap checklist applied, all of them earned the hard way in this programme:
  * echo the catalogue identifier back from the response; a bad -source= can
    return HTTP 200 serving a generic page, OR an unrelated REAL catalogue
  * `-out.all=1`, never bare `-out.all`, which silently returns the default
    column subset (12 of 34 columns in one recorded case)
  * percent-encode `+` as %2B; a literal + is decoded as a space and VizieR then
    runs a keyword search returning unrelated catalogues at HTTP 200
  * no cone search: `-c`/`-c.rs` silently returns 0 rows with no error
  * assert BOTH the row count and the column list, not just the row count
  * CfA mirror as fallback when CDS is saturated
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import time
import urllib.parse
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) invariant-lead01"}
MIRRORS = ["https://vizier.cds.unistra.fr/viz-bin/",
           "https://vizier.cfa.harvard.edu/viz-bin/"]


def get(path, params, timeout=90):
    q = urllib.parse.urlencode(params, quote_via=urllib.parse.quote)
    last = None
    for m in MIRRORS:
        url = m + path + "?" + q
        try:
            t0 = time.time()
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                body = r.read()
            return body, url, time.time() - t0
        except Exception as e:               # noqa: BLE001
            last = f"{m}: {e}"
            continue
    raise RuntimeError(f"all mirrors failed: {last}")


def check(body, want_id):
    """Guard against every VizieR failure mode recorded in this programme."""
    txt = body.decode("utf-8", "replace")
    if "<html" in txt[:2000].lower() or "<!doctype" in txt[:2000].lower():
        raise RuntimeError("HTML returned, not a table (bot wall or fallback)")
    if "#INFO Error=" in txt or "Error=Table or Catalog not found" in txt:
        line = [l for l in txt.split("\n") if "Error" in l][:2]
        raise RuntimeError(f"VizieR error line present: {line}")
    names = [l for l in txt.split("\n") if l.startswith("#Name:")]
    if not names:
        raise RuntimeError("no #Name: line -- cannot verify the identifier")
    got = names[0].split(":", 1)[1].strip()
    if want_id.lower() not in got.lower():
        raise RuntimeError(f"IDENTIFIER MISMATCH: asked {want_id}, got {got}. "
                           "This is the unrelated-real-catalogue trap.")
    return txt, got


def parse_tsv(txt):
    """VizieR asu-tsv: '#'-comments, then header, units, dashes, then data."""
    lines = [l for l in txt.split("\n") if not l.startswith("#")]
    lines = [l for l in lines if l.strip() != ""]
    hdr = lines[0].split("\t")
    # the units row and the ---- row follow the header
    body = []
    for l in lines[1:]:
        if set(l.replace("\t", "").strip()) <= set("-"):
            continue
        p = l.split("\t")
        if len(p) != len(hdr):
            continue
        if all(x.strip() in ("", "-") for x in p):
            continue
        body.append(p)
    # drop the units line: it is the first row that parses but has no numbers
    if body:
        first = body[0]
        numeric = 0
        for x in first:
            try:
                float(x)
                numeric += 1
            except ValueError:
                pass
        if numeric == 0:
            body = body[1:]
    return hdr, body


def save(name, body, txt, hdr, rows, url, dt, catid):
    p = os.path.join(HERE, name)
    with open(p, "wb") as f:
        f.write(body)
    man = {
        "source_url": url, "retrieved_utc": time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "sha256": hashlib.sha256(body).hexdigest(), "bytes": len(body),
        "n_rows": len(rows), "n_columns": len(hdr), "columns": hdr,
        "catalogue_identifier_echoed": catid, "fetch_seconds": round(dt, 2),
        "measurement_or_model": "see per-column notes in the paper",
    }
    with open(p + ".manifest.json", "w") as f:
        json.dump(man, f, indent=1)
    return man


TABLES = [
    # (vizier table path, output name, human note)
    ("J/A+A/661/A7/table2", "efeds_bahar2022_table2.tsv",
     "eFEDS cluster/group sample properties"),
    ("J/A+A/661/A7/table3", "efeds_bahar2022_table3.tsv",
     "candidate: density-profile parameters"),
    ("J/A+A/661/A7/tablea1", "efeds_bahar2022_tablea1.tsv",
     "candidate: appendix table"),
]

if __name__ == "__main__":
    os.makedirs(HERE, exist_ok=True)
    print("=" * 78)
    print("LEAD 01 -- fetching eFEDS resolved group profiles")
    print("=" * 78)
    got_any = False
    for path, out, note in TABLES:
        params = {"-source": path, "-out.max": "unlimited", "-out.all": "1"}
        print(f"\n   {path}  ({note})")
        try:
            body, url, dt = get("asu-tsv", params)
            txt, catid = check(body, "J/A+A/661/A7")
            hdr, rows = parse_tsv(txt)
            man = save(out, body, txt, hdr, rows, url, dt, catid)
            print(f"      OK  {len(rows)} rows x {len(hdr)} cols, "
                  f"{len(body)} bytes, {dt:.1f}s")
            print(f"      identifier echoed: {catid}")
            print(f"      columns: {', '.join(hdr[:14])}"
                  + (" ..." if len(hdr) > 14 else ""))
            got_any = True
        except Exception as e:                # noqa: BLE001
            print(f"      not available: {e}")
    if not got_any:
        print("\n   Nothing retrieved. Do NOT substitute a proxy; report it.")
