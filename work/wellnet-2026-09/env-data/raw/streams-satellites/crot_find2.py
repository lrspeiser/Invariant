"""Second-pass VizieR discovery: author-name search (descriptions embed 'Author+, YYYY')
plus direct bibcode resolution via the VizieR metadata endpoint."""
import sys, json
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from astroquery.vizier import Vizier
import requests

AUTHORS = ["Krajnovic", "Emsellem", "Cappellari", "Bevacqua", "Bershady",
           "Barrera-Ballesteros", "Bryant", "Moiseev", "Whitmore",
           "Corsini", "Bettoni", "Falcon-Barroso", "Serra", "Davis"]

out = {}
for a in AUTHORS:
    try:
        res = Vizier.find_catalogs(a, max_catalogs=200)
    except Exception as e:
        print("FAILED %r: %s" % (a, e)); continue
    print("\n=== author %r -> %d ===" % (a, len(res)))
    rows = []
    for k, v in res.items():
        if k is None:
            continue
        d = getattr(v, "description", "") or ""
        rows.append({"catalog": k, "description": d})
        print("  %-24s %s" % (k, d))
    out[a] = rows

# --- bibcode -> VizieR catalogue resolution ---
BIBCODES = {
    "Krajnovic2011_ATLAS3D_II":       "2011MNRAS.414.2923K",
    "Emsellem2011_ATLAS3D_III":       "2011MNRAS.414..888E",
    "Cappellari2011_ATLAS3D_I":       "2011MNRAS.413..813C",
    "Bevacqua2022_MaNGA_counterrot":  "2022MNRAS.511..139B",
    "Jin2016_MaNGA_misalign":         "2016MNRAS.463..913J",
    "Bryant2019_SAMI_misalign":       "2019MNRAS.483..458B",
    "BarreraBallesteros2014_CALIFA":  "2014A&A...568A..70B",
    "BarreraBallesteros2015_CALIFA":  "2015A&A...579A..45B",
    "Moiseev2011_SPRC":               "2011MNRAS.418..244M",
    "Whitmore1990_PRC":               "1990AJ....100.1489W",
    "Chen2016_MaNGA":                 "2016NatCo...713269C",
    "Duckworth2020":                  "2020MNRAS.492.1869D",
    "Serra2014_ATLAS3D_HI_misalign":  "2014MNRAS.444.3388S",
    "Davis2011_ATLAS3D_X":            "2011MNRAS.417..882D",
}
bib_out = {}
for name, bib in BIBCODES.items():
    url = ("https://vizier.cds.unistra.fr/viz-bin/VizieR?-source=&"
           "-bibcode=%s&-meta.all" % bib)
    try:
        r = requests.get(url, timeout=90,
                         headers={"User-Agent": "gravity-research-acquisition/1.0"})
        txt = r.text
    except Exception as e:
        print("BIB FAILED %s: %s" % (bib, e)); continue
    import re
    cats = sorted(set(re.findall(r"-source=([JIVB][^&\"'>\s]{4,40})", txt)))
    cats = [c for c in cats if "/" in c]
    print("BIB %-32s %-22s -> %s" % (name, bib, cats[:8]))
    bib_out[name] = {"bibcode": bib, "candidates": cats[:20], "url": url}

with open("crot_vizier_find2.json", "w", encoding="utf-8") as fh:
    json.dump({"authors": out, "bibcodes": bib_out}, fh, indent=2)
print("\nWROTE crot_vizier_find2.json")
