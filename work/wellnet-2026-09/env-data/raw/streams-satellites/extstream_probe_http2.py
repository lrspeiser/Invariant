"""VizieR probe with CORRECT URL ENCODING.

FAILURE MODE FOUND: '-source=J/A+A/671/A141' has its '+' decoded as a SPACE by
the query-string parser, so VizieR silently runs a generic catalogue search and
returns 468 kB of unrelated catalogues with HTTP 200 and NO error line.
The '+' MUST be percent-encoded as %2B.
"""
import sys, os, json
import requests

BASE = r"C:\Users\henry\Documents\Codex\2026-08-21\Invariant-main-integration\work\wellnet-2026-09\env-data\raw\streams-satellites"
sys.path.insert(0, BASE)

HDR = {"User-Agent": "gravity-research-acquisition/1.0 (academic data acquisition)"}


def enc(cat):
    return cat.replace("+", "%2B")


CANDIDATES = [
    "J/A+A/671/A141",   # SSLS I  Martinez-Delgado+2023
    "J/A+A/669/A103",
    "J/A+A/682/A22",
    "J/A+A/677/A80",
    "J/A+A/691/A73",
    "J/AJ/140/962",
    "J/ApJ/689/184",
    "J/ApJ/883/L32",
    "J/ApJ/705/1275",
    "J/ApJ/648/389",
    "J/ApJ/634/287",    # Ibata+2005?
    "J/MNRAS/351/117",  # Ibata+2004?
    "J/AJ/127/2723",    # Peng+2004 CenA PNe?
    "J/AJ/139/1871",    # Woodley+2010?
    "J/A+A/558/A42",
]


def probe(cat):
    url = "https://vizier.cds.unistra.fr/viz-bin/asu-tsv?-source=%s&-meta" % enc(cat)
    try:
        r = requests.get(url, timeout=120, headers=HDR)
        txt = r.text
    except Exception as e:
        return {"catalog": cat, "http_error": str(e), "url": url}
    err = [l for l in txt.splitlines() if l.startswith("#INFO") and "Error=" in l]
    tabs = [l for l in txt.splitlines() if l.startswith("#Table")]
    titles = [l for l in txt.splitlines() if l.startswith("#Title:")]
    return {"catalog": cat, "url": url, "status": r.status_code, "bytes": len(txt),
            "errors": err[:4], "tables": tabs[:60], "titles": titles[:20]}


out = {}
for cat in CANDIDATES:
    res = probe(cat)
    out[cat] = res
    print("=" * 72)
    print(cat, "http", res.get("status"), "bytes", res.get("bytes"),
          "ntitles", len(res.get("titles", [])))
    if res.get("errors"):
        print("   NOT FOUND:", res["errors"][0].strip())
    elif len(res.get("titles", [])) > 6:
        print("   SUSPECT: generic multi-catalogue response (probably bad encoding)")
        print("   ", res["titles"][0].strip())
    else:
        for t in res.get("titles", [])[:4]:
            print("   ", t.strip())
        for t in res.get("tables", []):
            print("   TABLE:", t.strip())

with open(os.path.join(BASE, "extstream_vizier_http_probe2.json"), "w", encoding="utf-8") as fh:
    json.dump(out, fh, indent=2)
print("\nwrote extstream_vizier_http_probe2.json")
