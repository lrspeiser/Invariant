"""Third-pass VizieR discovery: targeted terms for misalignment / counter-rotation /
polar rings, plus more author names."""
import sys, json
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from astroquery.vizier import Vizier

TERMS = [
    "counterrotation", "counter-rotating", "counterrotating",
    "polar ring", "polar disc", "polar disk",
    "kinematic misalignment", "misalignment angle galaxies",
    "kinematically decoupled", "kinematic position angle",
    "ionized gas stellar kinematics disc galaxies",
    "MaNGA DynPop circular velocity",
    "SAMI galaxy survey data release", "CALIFA survey data release",
    "MaNGA Pipe3D kinematics", "lambda_R spin parameter",
]
AUTHORS = ["Jin", "Chen", "Xu", "Bao", "Zhou", "Duckworth", "Li", "Zhu",
           "Gasymov", "Combes", "Pizzella", "Vega Beltran", "Sarzi",
           "Mendez-Abreu", "Katkov", "Silchenko", "Bizyaev", "Reshetnikov",
           "van de Voort", "Raimundo", "Beom", "Ristea"]

KEY = ["counter", "counterrot", "polar", "misalign", "kinemat", "manga",
       "califa", "sami", "atlas3d", "ring", "decoupl", "rotation", "spin",
       "velocity", "dynpop"]

out = {}
for q in TERMS + AUTHORS:
    try:
        res = Vizier.find_catalogs(q, max_catalogs=300)
    except Exception as e:
        print("FAILED %r: %s" % (q, e)); continue
    rows = []
    for k, v in res.items():
        if k is None:
            continue
        d = getattr(v, "description", "") or ""
        rows.append({"catalog": k, "description": d})
    out[q] = rows
    hits = [r for r in rows if any(w in (r["description"] or "").lower() for w in KEY)]
    print("\n=== %r  (%d total, %d keyword hits) ===" % (q, len(rows), len(hits)))
    for r in hits[:40]:
        print("   %-24s %s" % (r["catalog"], r["description"]))

with open("crot_vizier_find3.json", "w", encoding="utf-8") as fh:
    json.dump(out, fh, indent=2)
print("\nWROTE crot_vizier_find3.json")
