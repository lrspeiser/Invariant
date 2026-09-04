"""Probe VizieR for external-galaxy stream catalogues: list REAL table names.

Do not guess table1/table2 -- ask VizieR what tables a catalogue actually has.
"""
import sys, os, json, traceback

BASE = r"C:\Users\henry\Documents\Codex\2026-08-21\Invariant-main-integration\work\wellnet-2026-09\env-data\raw\streams-satellites"
sys.path.insert(0, BASE)

from astroquery.vizier import Vizier

CANDIDATES = [
    "J/A+A/671/A141",   # Martinez-Delgado+2023 SSLS I
    "J/AJ/140/962",     # Martinez-Delgado+2010 Local Volume tidal streams
    "J/A+A/669/A103",   # guess: Miro-Carretero SSLS II?
    "J/A+A/682/A22",    # guess
    "J/ApJ/689/184",    # Martinez-Delgado+2008 NGC5907
    "J/ApJ/883/L32",    # van Dokkum+2019 Dragonfly NGC5907
    "J/ApJ/705/1275",   # Gilbert+2009 M31 GSS kinematics
    "J/ApJ/648/389",    # Kalirai+2006
    "J/AJ/128/237",     # Peng+2004 CenA PNe?
    "J/AJ/139/1871",    # Woodley+2010 CenA GCs?
    "J/MNRAS/408/L109",
]

out = {}
for cat in CANDIDATES:
    rec = {"catalog": cat}
    try:
        cats = Vizier.get_catalogs(cat)
        rec["ok"] = True
        rec["tables"] = []
        for t in cats:
            rec["tables"].append({
                "name": t.meta.get("name"),
                "description": str(t.meta.get("description"))[:300],
                "nrows_preview": len(t),
                "columns": [c for c in t.colnames],
            })
    except Exception as e:
        rec["ok"] = False
        rec["error"] = "%s: %s" % (type(e).__name__, e)
    out[cat] = rec
    print("=" * 70)
    print(cat, "OK" if rec.get("ok") else "FAIL")
    if rec.get("ok"):
        for t in rec["tables"]:
            print("   table:", t["name"], "| preview rows:", t["nrows_preview"])
            print("      desc:", t["description"][:180])
            print("      cols:", t["columns"])
    else:
        print("   ", rec.get("error"))

with open(os.path.join(BASE, "extstream_vizier_probe.json"), "w", encoding="utf-8") as fh:
    json.dump(out, fh, indent=2)
print("\nwrote extstream_vizier_probe.json")
