"""Fast direct-HTTP probe of VizieR catalogue existence + real table names.

Uses the CDS ReadMe endpoint and the asu-tsv -meta endpoint.
"""
import sys, os, json
import requests

BASE = r"C:\Users\henry\Documents\Codex\2026-08-21\Invariant-main-integration\work\wellnet-2026-09\env-data\raw\streams-satellites"
sys.path.insert(0, BASE)

HDR = {"User-Agent": "gravity-research-acquisition/1.0 (academic data acquisition)"}

CANDIDATES = [
    "J/A+A/671/A141",
    "J/AJ/140/962",
    "J/ApJ/689/184",
    "J/ApJ/883/L32",
    "J/ApJ/705/1275",
    "J/ApJ/648/389",
    "J/A+A/682/A22",
    "J/A+A/669/A103",
]


def probe(cat):
    url = "https://vizier.cds.unistra.fr/viz-bin/asu-tsv?-source=%s&-meta" % cat
    try:
        r = requests.get(url, timeout=90, headers=HDR)
        txt = r.text
    except Exception as e:
        return {"catalog": cat, "http_error": str(e)}
    err = [l for l in txt.splitlines() if l.startswith("#INFO") and "Error=" in l]
    tabs = [l for l in txt.splitlines() if l.startswith("#Table")]
    names = [l for l in txt.splitlines() if l.startswith("#Name:")]
    titles = [l for l in txt.splitlines() if l.startswith("#Title:")]
    return {
        "catalog": cat,
        "status": r.status_code,
        "bytes": len(txt),
        "errors": err[:4],
        "tables": tabs[:40],
        "names": names[:40],
        "titles": titles[:12],
        "head": txt[:400],
    }


out = {}
for cat in CANDIDATES:
    res = probe(cat)
    out[cat] = res
    print("=" * 72)
    print(cat, "http", res.get("status"), "bytes", res.get("bytes"))
    if res.get("errors"):
        print("   ERROR LINES:", res["errors"])
    else:
        for t in res.get("titles", [])[:4]:
            print("   ", t.strip())
        for t in res.get("tables", []):
            print("   TABLE:", t.strip())

with open(os.path.join(BASE, "extstream_vizier_http_probe.json"), "w", encoding="utf-8") as fh:
    json.dump(out, fh, indent=2)
print("\nwrote extstream_vizier_http_probe.json")
