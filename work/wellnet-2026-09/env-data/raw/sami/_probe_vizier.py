"""Record the VizieR NEGATIVE result explicitly, with the guard the brief asks
for: VizieR answers HTTP 200 with a generic page for a nonexistent -source=,
so a hit is only a hit if the body carries '#Table', '#Column' AND echoes the
identifier back.
"""
import hashlib
import json
import os
from datetime import datetime, timezone

import requests
from astroquery.vizier import Vizier

OUT = os.path.dirname(os.path.abspath(__file__))
BASE = "https://vizier.cds.unistra.fr/viz-bin/asu-tsv?-source=%s&-out.max=5&-out.all"

# The SAMI DR3 paper is MNRAS 505, 991; Owers+2017 is MNRAS 468, 1824;
# Bryant+2015 is MNRAS 447, 2857.  Neighbouring page numbers are swept because
# CDS occasionally indexes a catalogue under an adjacent first page.
CANDIDATES = [
    "J/MNRAS/505/991", "J/MNRAS/505/990", "J/MNRAS/505/992", "J/MNRAS/505/1",
    "J/MNRAS/468/1824", "J/MNRAS/468/1823", "J/MNRAS/468/1825",
    "J/MNRAS/447/2857", "J/MNRAS/447/2856", "J/MNRAS/447/2858",
    "J/ApJ/873/52", "J/ApJ/873/51", "J/ApJ/873/53",
    "J/MNRAS/446/1567",   # SAMI Early Data Release (Allen+2015) - expected HIT
]


def main():
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    res = []
    for cid in CANDIDATES:
        url = BASE % cid
        r = requests.get(url, timeout=180)
        body = r.text
        hit = ("#Table" in body and "#Column" in body and cid in body
               and "Table or Catalog not found" not in body)
        err = None
        if not hit:
            for line in body.splitlines():
                if "Error=" in line:
                    err = line.split("Error=")[1].strip()
                    break
        res.append({"catalogue": cid, "url": url, "http_status": r.status_code,
                    "bytes": len(r.content), "is_real_hit": hit,
                    "vizier_error": err})
        print("%-20s HTTP %s  hit=%-5s  %s" % (cid, r.status_code, hit, err or ""))

    meta = []
    for term in ["SAMI Galaxy Survey", "SAMI", "Owers", "Croom SAMI cluster"]:
        try:
            cats = Vizier.find_catalogs(term)
            meta.append({"term": term,
                         "matches": [{"id": k, "description": v.description}
                                     for k, v in cats.items()]})
        except Exception as e:
            meta.append({"term": term, "error": repr(e)})

    doc = {
        "purpose": "Negative-result record: VizieR does NOT hold the SAMI DR3 "
                   "catalogues or the Owers et al. 2017 cluster catalogue.",
        "retrieved_utc": ts,
        "service": "https://vizier.cds.unistra.fr/ (asu-tsv + find_catalogs metadata search)",
        "guard": "A response counts as a hit only if the body contains '#Table' and "
                 "'#Column', echoes the requested identifier, and does not say "
                 "'Table or Catalog not found'. VizieR returns HTTP 200 for "
                 "nonexistent sources.",
        "direct_probes": res,
        "metadata_searches": meta,
        "conclusion": (
            "The only SAMI holding in VizieR is J/MNRAS/446/1567, the 2015 SAMI "
            "Early Data Release (Allen et al. 2015, 107 galaxies), which is "
            "superseded by DR3 and does not cover the cluster regions. Neither "
            "SAMI DR3 (Croom et al. 2021, MNRAS 505, 991) nor Owers et al. 2017 "
            "(MNRAS 468, 1824) nor Bryant et al. 2015 (MNRAS 447, 2857) nor Owers "
            "et al. 2019 (ApJ 873, 52) has a VizieR catalogue. The Data Central "
            "IVOA TAP service and the arXiv LaTeX sources are the only routes."
        ),
    }
    p = os.path.join(OUT, "vizier_negative_probe.json")
    json.dump(doc, open(p, "w"), indent=1)
    h = hashlib.sha256(open(p, "rb").read()).hexdigest()
    json.dump({"file": "vizier_negative_probe.json",
               "kind": "provenance record of a negative search, not a data file",
               "source_url": "https://vizier.cds.unistra.fr/viz-bin/asu-tsv",
               "retrieved_utc": ts, "sha256": h, "bytes": os.path.getsize(p),
               "row_count": len(res), "column_count": 6,
               "columns": [{"name": k, "unit": "-"} for k in
                           ["catalogue", "url", "http_status", "bytes",
                            "is_real_hit", "vizier_error"]],
               "query": "GET " + BASE % "<CAT>" + " for each candidate identifier, "
                        "plus astroquery Vizier.find_catalogs on four search terms"},
              open(p + ".manifest.json", "w"), indent=1)
    print("\nwrote", p)


if __name__ == "__main__":
    main()
