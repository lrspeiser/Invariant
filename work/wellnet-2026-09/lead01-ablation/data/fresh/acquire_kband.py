#!/usr/bin/env python3
"""
Acquire 2MASS K_s TOTAL (extrapolated) magnitudes for the 94 Babyk+2018 ETGs.

ACQUISITION ONLY. No mass-to-light ratio is applied. No mass is computed.
No residuals, no acceleration ratios, no model comparison.

Route: Sesame name resolution -> VizieR VII/233/xsc cone search ->
       column K.ext (2MASS k_m_ext, the total extrapolated K_s magnitude).
Fallbacks recorded per object.

TRAP: the VizieR ASU error marker is '#INFO' + TAB + 'Error='.
Testing for the string '#INFO Error=' with a SPACE silently never matches,
turning a missing catalogue into a false 'exists'. We test for '\\tError='.
"""
import hashlib
import json
import os
import re
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone

BASE = os.path.dirname(os.path.abspath(__file__))
RAWK = os.path.join(BASE, "raw", "kband")
os.makedirs(RAWK, exist_ok=True)

XSC = "VII/233/xsc"
CONE_ARCSEC = 60
UA = {"User-Agent": "curl/8 (acquisition; astrophysics data ingest)"}


def fetch(url, cache_name, timeout=90, retries=3):
    """GET with on-disk cache of the RAW unmodified response body."""
    p = os.path.join(RAWK, cache_name)
    if os.path.exists(p) and os.path.getsize(p) > 0:
        with open(p, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    last = None
    for a in range(retries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                body = r.read().decode("utf-8", errors="replace")
            with open(p, "w", encoding="utf-8") as f:
                f.write(body)
            time.sleep(0.25)
            return body
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(1.5 * (a + 1))
    print(f"    FETCH FAILED {cache_name}: {last}")
    return None


def vizier_error(body):
    """The ONLY reliable existence/error test: '#INFO' TAB 'Error='."""
    return body is None or "\tError=" in body


# ---------------------------------------------------------------- Sesame ----
def sesame(name):
    """Resolve a name to (ra_deg, dec_deg, service_used). Raw body cached."""
    safe = urllib.parse.quote(name)
    for host, tag in (("https://cdsweb.u-strasbg.fr", "stras"),
                      ("https://vizier.cfa.harvard.edu", "cfa")):
        url = f"{host}/cgi-bin/nph-sesame/-oI?{safe}"
        body = fetch(url, f"sesame_{tag}_{name}.txt")
        if body is None:
            continue
        m = re.search(r"^%J\s+([-+0-9.]+)\s+([-+0-9.]+)", body, re.M)
        if m:
            return float(m.group(1)), float(m.group(2)), tag, url
    return None, None, None, None


# ---------------------------------------------------------------- VizieR ----
def parse_asu_tsv(body):
    """Parse a VizieR asu-tsv body into (colnames, units, [rows as dicts])."""
    lines = body.split("\n")
    hdr_i = None
    for i, ln in enumerate(lines):
        if ln.startswith("#") or ln.strip() == "":
            continue
        hdr_i = i
        break
    if hdr_i is None:
        return [], [], []
    cols = lines[hdr_i].split("\t")
    units = lines[hdr_i + 1].split("\t") if hdr_i + 1 < len(lines) else []
    # line hdr_i+2 is the ---- rule
    rows = []
    for ln in lines[hdr_i + 3:]:
        if ln.startswith("#") or ln.strip() == "":
            continue
        cells = ln.split("\t")
        if len(cells) < 2:
            continue
        rows.append(dict(zip(cols, cells)))
    return cols, units, rows


def f(v):
    v = (v or "").strip()
    if v == "":
        return None
    try:
        return float(v)
    except ValueError:
        return None


def xsc_cone(name, ra, dec):
    q = (f"https://vizier.cfa.harvard.edu/viz-bin/asu-tsv?-source={XSC}"
         f"&-c={ra:+.6f}{dec:+.6f}&-c.rs={CONE_ARCSEC}"
         f"&-out.max=50&-out.all&-sort=_r")
    body = fetch(q, f"xsc_{name}.tsv")
    if vizier_error(body):
        return None, q, body
    _, _, rows = parse_asu_tsv(body)
    if not rows:
        return None, q, body
    # nearest first (sorted by _r); pick the nearest row that HAS a K.ext
    best = None
    for r in rows:
        if f(r.get("K.ext")) is not None:
            best = r
            break
    if best is None:
        best = rows[0]
    return best, q, body


# ------------------------------------------------------------------ main ----
names = [l.strip() for l in
         open(os.path.join(BASE, "babyk2018_joined_per_object.tsv"),
              encoding="utf-8").read().split("\n")[1:] if l.strip()]
names = [l.split("\t")[0] for l in names]
assert len(names) == 94, f"expected 94 Babyk names, got {len(names)}"
print(f"{len(names)} Babyk names loaded; first={names[0]} last={names[-1]}")

results = []
for i, nm in enumerate(names, 1):
    ra, dec, svc, surl = sesame(nm)
    if ra is None:
        print(f"[{i:3d}/94] {nm:12s} SESAME FAILED")
        results.append(dict(name=nm, ra=None, dec=None, sesame=None,
                            sep=None, kext=None, ekext=None, row=None,
                            flag="sesame_unresolved", query=None))
        continue
    row, q, _ = xsc_cone(nm, ra, dec)
    if row is None:
        print(f"[{i:3d}/94] {nm:12s} resolved({svc}) but NO XSC MATCH")
        results.append(dict(name=nm, ra=ra, dec=dec, sesame=svc, sep=None,
                            kext=None, ekext=None, row=None,
                            flag="no_xsc_match", query=q))
        continue
    sep = f(row.get("_r"))
    kext, ekext = f(row.get("K.ext")), f(row.get("e_K.ext"))
    flag = "" if kext is not None else "xsc_match_no_Kext"
    results.append(dict(name=nm, ra=ra, dec=dec, sesame=svc, sep=sep,
                        kext=kext, ekext=ekext, row=row, flag=flag,
                        query=q))
    print(f"[{i:3d}/94] {nm:12s} sep={sep if sep is not None else -1:6.2f}\" "
          f"K.ext={kext if kext is not None else float('nan'):7.3f} "
          f"2MASX={row.get('2MASX','')[:17]} {flag}")

with open(os.path.join(BASE, "_kband_stage1.json"), "w", encoding="utf-8") as fh:
    json.dump([{k: v for k, v in r.items() if k != "row"} |
               {"row": {kk: r["row"][kk] for kk in
                        ("_r", "2MASX", "RAJ2000", "DEJ2000", "K.ext",
                         "e_K.ext", "r.ext", "K.K20e", "e_K.K20e", "n_K.ext")
                        if r["row"] and kk in r["row"]} if r["row"] else None}
              for r in results], fh, indent=2)

ok = [r for r in results if r["kext"] is not None]
bad = [r for r in results if r["kext"] is None]
print(f"\nSTAGE 1: {len(ok)}/94 with K.ext; {len(bad)} need a fallback")
for r in bad:
    print("   NEEDS FALLBACK:", r["name"], r["flag"])
