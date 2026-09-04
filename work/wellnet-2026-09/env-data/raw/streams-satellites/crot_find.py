"""VizieR catalogue discovery for the counter-rotator / polar-disc lane.
Uses Vizier.find_catalogs (metadata search) rather than guessing -source= ids.
"""
import sys, io, json
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from astroquery.vizier import Vizier

QUERIES = [
    "ATLAS3D kinematic classification early-type galaxies",
    "ATLAS3D lambda_R slow fast rotators",
    "ATLAS3D volume-limited sample early-type",
    "kinematically decoupled core counter-rotating",
    "MaNGA counter-rotating galaxies",
    "MaNGA kinematic misalignment gas stellar",
    "CALIFA kinematic misalignment",
    "SAMI kinematic misalignment",
    "polar ring galaxies catalogue",
    "SDSS-based polar ring catalogue",
    "gas-star counter-rotation",
    "stellar gas kinematic position angle misalignment",
]

out = {}
for q in QUERIES:
    try:
        res = Vizier.find_catalogs(q, max_catalogs=40)
    except Exception as e:
        print("QUERY FAILED %r: %s" % (q, e)); continue
    print("\n=== %r -> %d ===" % (q, len(res)))
    rows = []
    for k, v in res.items():
        desc = getattr(v, "description", "")
        print("  %-28s %s" % (k, desc))
        rows.append({"catalog": k, "description": desc})
    out[q] = rows

with open("crot_vizier_find_catalogs.json", "w", encoding="utf-8") as fh:
    json.dump(out, fh, indent=2)
print("\nWROTE crot_vizier_find_catalogs.json")
